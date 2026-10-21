"""SA-V2X: a signal-aware V2X trajectory and crosswalk-arrival predictor.

One network serves both placements (RSU-side and vehicle-side); the case and
ego flags enter as agent attributes, and the vehicle-side model is fine-tuned
from the RSU-side weights.

    history (SDSM / BSM / self)  -> AgentEncoder (per-step MLP + GRU)
    MAP lanes, connectors, crosswalks -> MapEncoder (PointNet over polylines)
    SPaT per signal group        -> SignalEncoder, then *bound* to the connector
                                    and crosswalk tokens it controls (MAP signalGroup)
    interaction                  -> 2 x [agent->(map+signal) attention, agent->agent
                                    attention], keys carry agent-relative geometry
    heads (agent-centric frame)  -> maneuver (scores the approach's left/straight/
                                    right connectors), K-mode Laplace trajectories,
                                    discrete-time hazards for target-crosswalk
                                    entry and exit (censoring-aware)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.features import ATTR_F, BSM_F, HIST_F, SIG_F, SCALE_XY


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def rel_geom(pa, pb):
    """Geometry of poses pb seen from poses pa. pa (...,3), pb (...,3) -> (...,6)."""
    dx, dy = pb[..., 0] - pa[..., 0], pb[..., 1] - pa[..., 1]
    c, s = torch.cos(pa[..., 2]), torch.sin(pa[..., 2])
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    dist = torch.sqrt(lx ** 2 + ly ** 2 + 1e-6)
    dh = pb[..., 2] - pa[..., 2]
    return torch.stack([lx / SCALE_XY, ly / SCALE_XY, dist / SCALE_XY, torch.cos(dh), torch.sin(dh),
                        torch.exp(-dist / 10.0)], dim=-1)


class RelAttention(nn.Module):
    """Each query attends over its own key set; keys = token + relative-geometry embedding."""

    def __init__(self, d, heads=4, drop=0.0):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.rel = mlp(6, d, d)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.ff = mlp(d, 2 * d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, ctx, geom, mask, bias=None):
        # x (B,A,d) ctx (B,A,N,d) or (B,1,N,d); geom (B,A,N,6); mask (B,A,N) bool (True = attend)
        B, A, d = x.shape
        N = geom.shape[2]
        h = self.h
        kv = ctx + self.rel(geom)
        q = self.q(self.n1(x)).view(B, A, h, d // h)
        k = self.k(kv).view(B, A, N, h, d // h)
        v = self.v(kv).view(B, A, N, h, d // h)
        att = torch.einsum("bahd,banhd->bahn", q, k) / math.sqrt(d // h)
        if bias is not None:
            att = att + bias                                            # (B,A,h,N)
        att = att.masked_fill(~mask.unsqueeze(2), -1e4)
        w = torch.softmax(att, dim=-1) * mask.unsqueeze(2).any(-1, keepdim=True)
        out = torch.einsum("bahn,banhd->bahd", w, v).reshape(B, A, d)
        x = x + self.drop(self.o(out))
        return x + self.drop(self.ff(self.n2(x)))


class SAV2X(nn.Module):
    def __init__(self, n_sg, n_bins=40, n_fut=40, d=64, K=6, map_attr=16, use_signals=True,
                 map_radius_m=80.0, agent_radius_m=60.0):
        super().__init__()
        self.K, self.T, self.nb, self.d = K, n_fut, n_bins, d
        self.use_signals = use_signals
        self.map_r, self.agent_r = map_radius_m, agent_radius_m
        # agent encoder
        self.step = mlp(HIST_F + BSM_F + 1, d, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.attr = mlp(ATTR_F, d, d)
        self.agent_out = mlp(2 * d, d, d)
        # map encoder
        self.pt = mlp(4, d, d)
        self.map_attr = mlp(map_attr, d, d)
        self.map_out = mlp(2 * d, d, d)
        # signals
        self.sig = mlp(SIG_F + n_bins + 1, d, d)
        self.sig_id = nn.Embedding(n_sg, d)
        self.no_sig = nn.Parameter(torch.zeros(d))
        self.bind = nn.Linear(d, d)
        # interaction
        self.ctx_layers = nn.ModuleList([RelAttention(d) for _ in range(2)])
        self.soc_layers = nn.ModuleList([RelAttention(d) for _ in range(2)])
        self.same_appr_bias = nn.Parameter(torch.zeros(4))
        # heads
        self.man_head = mlp(3 * d + 6, d, 1)
        self.modes = nn.Parameter(torch.randn(K, d) * 0.1)
        self.man_emb = nn.Linear(3, d)
        self.traj_head = mlp(2 * d, 2 * d, n_fut * 4)
        self.mode_head = mlp(2 * d, d, 1)
        self.haz_head = mlp(4 * d + 6, 2 * d, 2 * n_bins)

    # ------------------------------------------------------------------
    def encode_map(self, mt):
        pts = mt["pts"]                                                 # (M,P,2)
        cen = pts.mean(1, keepdim=True)
        dirs = torch.diff(pts, dim=1, prepend=pts[:, :1])
        f = torch.cat([(pts - cen) / 20.0, dirs], -1)
        p = self.pt(f).max(1).values
        return self.map_out(torch.cat([p, self.map_attr(mt["attr"])], -1))  # (M,d)

    def encode_signals(self, sig, sig_mask):
        # sig (B,G,F) sig_mask (B,G)
        B, G, _ = sig.shape
        ids = self.sig_id.weight.unsqueeze(0).expand(B, G, -1)
        if not self.use_signals:
            return ids + self.no_sig, torch.zeros_like(sig_mask)
        e = self.sig(torch.cat([sig, sig_mask.unsqueeze(-1)], -1)) + ids
        e = torch.where(sig_mask.unsqueeze(-1) > 0, e, ids + self.no_sig)
        return e, sig_mask

    def forward(self, b, mt):
        hist, bsm, attr, pose = b["hist"], b["bsm"], b["attr"], b["pose"]
        B, A, H, _ = hist.shape
        amask = b["amask"]                                              # (B,A) bool
        # ---- agents
        has_bsm = (bsm.abs().sum(-1, keepdim=True) > 0).float()
        x = self.step(torch.cat([hist, bsm, has_bsm], -1)).view(B * A, H, -1)
        _, hN = self.gru(x)
        a = self.agent_out(torch.cat([hN[-1].view(B, A, -1), self.attr(attr)], -1))
        # ---- map + signals, with signal binding into connector / crosswalk tokens
        m = self.encode_map(mt)                                         # (M,d)
        s, smask = self.encode_signals(b["sig"], b["sig_mask"])         # (B,G,d)
        msg = mt["sg"]                                                  # (M,)
        bound = torch.zeros(B, m.shape[0], self.d, device=m.device)
        has = msg >= 0
        bound[:, has] = self.bind(s[:, msg[has]])
        mtok = m.unsqueeze(0) + bound                                   # (B,M,d)
        M, G = mtok.shape[1], s.shape[1]
        # signal tokens get the pose of the first token they control (fallback: origin)
        sg_pose = torch.zeros(G, 3, device=m.device)
        for g in range(G):
            idx = (msg == g).nonzero()
            if len(idx):
                sg_pose[g] = mt["pose"][idx[0, 0]]
        ctx_pose = torch.cat([mt["pose"], sg_pose], 0)                  # (M+G,3)
        ctx = torch.cat([mtok, s], 1)                                   # (B,M+G,d)
        geom_c = rel_geom(pose.unsqueeze(2), ctx_pose.view(1, 1, M + G, 3))   # (B,A,M+G,6)
        dist_c = geom_c[..., 2] * SCALE_XY
        cmask = torch.cat([dist_c[..., :M] < self.map_r, smask.unsqueeze(1).expand(B, A, G) > 0], -1)
        cmask = cmask | (torch.arange(M + G, device=m.device) == mt["target_tok"]).view(1, 1, -1)
        geom_a = rel_geom(pose.unsqueeze(2), pose.unsqueeze(1))         # (B,A,A,6)
        smask_a = (geom_a[..., 2] * SCALE_XY < self.agent_r) & amask.unsqueeze(1) & amask.unsqueeze(2)
        appr = b["appr"]
        same = (appr.unsqueeze(1) == appr.unsqueeze(2)) & (appr.unsqueeze(1) >= 0)
        bias = same.unsqueeze(-1).float() * self.same_appr_bias.view(1, 1, 1, -1)   # (B,A,A,h)
        bias = bias.permute(0, 1, 3, 2)
        for lc, ls in zip(self.ctx_layers, self.soc_layers):
            a = lc(a, ctx.unsqueeze(1).expand(B, A, M + G, self.d), geom_c, cmask)
            a = ls(a, a.unsqueeze(1).expand(B, A, A, self.d), geom_a, smask_a, bias=bias)
        # ---- maneuver head: score the approach's left / straight / right connectors
        mtk = mt["man_tok"]                                             # (4,3,2)
        ap = appr.clamp(min=0)
        tok = mtk[ap]                                                   # (B,A,3,2)
        tokc = tok.clamp(min=0)
        emb = mtok[torch.arange(B).view(B, 1, 1, 1), tokc]              # (B,A,3,2,d)
        valid_t = (tok >= 0).float().unsqueeze(-1)
        emb = (emb * valid_t).sum(3) / valid_t.sum(3).clamp(min=1)      # (B,A,3,d)
        tpose = mt["pose"][tokc[..., 0]]                                # (B,A,3,3)
        g_m = rel_geom(pose.unsqueeze(2), tpose)
        man_logit = self.man_head(torch.cat([a.unsqueeze(2).expand(-1, -1, 3, -1), emb,
                                             emb * a.unsqueeze(2), g_m], -1)).squeeze(-1)
        man_ok = (tok[..., 0] >= 0) & (appr >= 0).unsqueeze(-1)
        man_logit = man_logit.masked_fill(~man_ok, -1e4)
        man_p = torch.where((appr >= 0).unsqueeze(-1), torch.softmax(man_logit, -1),
                            torch.zeros_like(man_logit))
        # ---- trajectories
        q = a.unsqueeze(2) + self.modes.view(1, 1, self.K, -1)
        mctx = self.man_emb(man_p).unsqueeze(2).expand(-1, -1, self.K, -1)
        hq = torch.cat([q, mctx], -1)
        out = self.traj_head(hq).view(B, A, self.K, self.T, 4)
        loc = torch.cumsum(out[..., :2], dim=3) * 2.0                   # meters, agent frame
        scale = F.softplus(out[..., 2:]) + 0.05
        mode_logit = self.mode_head(hq).squeeze(-1)
        # ---- crosswalk hazards
        tt = mt["target_tok"]
        cw = mtok[:, tt].unsqueeze(1).expand(-1, A, -1)
        g_cw = rel_geom(pose, mt["pose"][tt].view(1, 1, 3).expand(B, A, 3))
        ped_sg = s[:, mt["target_sg_idx"]].unsqueeze(1).expand(-1, A, -1)
        msg_idx = mt["man_sg"][ap]                                      # (B,A,3)
        mv_sg = s[torch.arange(B).view(B, 1, 1), msg_idx.clamp(min=0)]  # (B,A,3,d)
        mv_sg = (mv_sg * (man_p * (msg_idx >= 0)).unsqueeze(-1)).sum(2)
        haz = self.haz_head(torch.cat([a, cw, ped_sg, mv_sg, g_cw], -1))
        return {"loc": loc, "scale": scale, "mode_logit": mode_logit, "man_logit": man_logit,
                "man_ok": man_ok, "entry_logit": haz[..., :self.nb], "exit_logit": haz[..., self.nb:]}


# ---------------------------------------------------------------------- utilities
def to_agent_frame(xy, pose):
    """xy (...,T,2) world -> agent frame of pose (...,3)."""
    c, s = torch.cos(pose[..., 2]).unsqueeze(-1), torch.sin(pose[..., 2]).unsqueeze(-1)
    dx, dy = xy[..., 0] - pose[..., :1], xy[..., 1] - pose[..., 1:2]
    return torch.stack([c * dx + s * dy, -s * dx + c * dy], -1)


def to_world(loc, pose):
    """loc (B,A,K,T,2) agent frame -> world."""
    c = torch.cos(pose[..., 2]).view(*pose.shape[:-1], 1, 1)
    s = torch.sin(pose[..., 2]).view(*pose.shape[:-1], 1, 1)
    x = c * loc[..., 0] - s * loc[..., 1] + pose[..., 0].view(*pose.shape[:-1], 1, 1)
    y = s * loc[..., 0] + c * loc[..., 1] + pose[..., 1].view(*pose.shape[:-1], 1, 1)
    return torch.stack([x, y], -1)


def hazard_pmf(logit):
    """Discrete-time hazards -> (pmf over bins, survival beyond the last bin)."""
    h = torch.sigmoid(logit)
    surv = torch.cumprod(1 - h, -1)
    prev = torch.cat([torch.ones_like(surv[..., :1]), surv[..., :-1]], -1)
    return prev * h, surv[..., -1]
