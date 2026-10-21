"""SA-V2X-Ego: vehicle-side, signal-aware trajectory and crosswalk-conflict predictor.

Inputs (all in the ego frame at t0):
    ego BSM history (3 s)          -> step MLP + GRU
    ped presence by zone (3 s)     -> step MLP + GRU         (SDSM, presence only)
    context                        -> MLP                    (lane, link status, leader BSM)
    candidate routes (L / S / R)   -> path MLP + route scalars, each bound to its
                                      signal group's SPaT token
    pedestrian signal group        -> SPaT token
Fusion, per route r:  z_r = MLP([ctx, route_r, ctx * route_r, ped signal])
Heads:
    maneuver      logit per legal route
    trajectory    2 modes per route, decoded along the route: progress s(t) from
                  non-negative speeds (so stopping at a red light is a plateau) plus a
                  lateral offset from the path; Laplace scales on x and y.
                  6 modes in all, probability P(route) * P(mode | route)
    arrival       discrete-time hazards for entering and leaving the target crosswalk
                  (0.5 s bins to 20 s), per route; routes that do not cross it never enter
    occupancy     probability that a pedestrian occupies each crosswalk half, per bin
    conflict      probability of a conflict (same half within theta), per route
Per-route outputs are combined with P(route) at inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sav2x.features import F_CTX, F_HIST, F_PRES, F_ROUTE, F_SIG, NB, NP, T

PATH_SPACING = 4.0
DT_FUT = 0.2


def mlp(i, h, o, n=2, drop=0.0):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
        if drop:
            layers.append(nn.Dropout(drop))
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class EgoNet(nn.Module):
    def __init__(self, d=96, modes_per_route=2, drop=0.1):
        super().__init__()
        self.d, self.M = d, modes_per_route
        self.step = mlp(F_HIST, d, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.pstep = mlp(F_PRES, 48, 48)
        self.pgru = nn.GRU(48, d // 2, batch_first=True)
        self.ctx_in = mlp(F_CTX, d, d)
        self.ctx = mlp(d + d // 2 + d, 2 * d, d, drop=drop)
        self.sig = mlp(F_SIG + 1, d, d)
        self.sig_type = nn.Embedding(2, d)              # 0 route group, 1 pedestrian group
        self.no_sig = nn.Parameter(torch.zeros(d))
        self.path = mlp(NP * 2, 2 * d, d)
        self.rf = mlp(F_ROUTE, d, d)
        self.route = mlp(3 * d, 2 * d, d)
        self.fuse = mlp(4 * d + d // 2, 2 * d, d, n=3, drop=drop)
        self.man = nn.Linear(d, 1)
        self.mode_emb = nn.Parameter(torch.randn(modes_per_route, d) * 0.1)
        self.traj = mlp(d, 2 * d, T * 4)
        self.mode_logit = nn.Linear(d, 1)
        self.haz = mlp(2 * d + d // 2, 2 * d, 2 * NB)
        self.conf = mlp(2 * d + d // 2, d, 1)
        self.occ = mlp(d + d // 2 + d, d, 2 * NB)

    def encode_sig(self, sig, smask):
        B = sig.shape[0]
        typ = torch.tensor([0, 0, 0, 1], device=sig.device)
        e = self.sig(torch.cat([sig, smask.unsqueeze(-1)], -1)) + self.sig_type(typ)
        return torch.where(smask.unsqueeze(-1) > 0, e, self.sig_type(typ).expand(B, -1, -1) + self.no_sig)

    def forward(self, b):
        B = b["hist"].shape[0]
        _, h = self.gru(self.step(b["hist"]))
        h = h[-1]
        _, p = self.pgru(self.pstep(b["pres"]))
        p = p[-1]
        ctx = self.ctx(torch.cat([h, p, self.ctx_in(b["ctx"])], -1))                      # (B,d)
        s = self.encode_sig(b["sig"], b["smask"])                                          # (B,4,d)
        s_route, s_ped = s[:, :3], s[:, 3]
        pth = b["path"] / 50.0
        r = self.route(torch.cat([self.path(pth.flatten(2)), self.rf(b["rfeat"]), s_route], -1))   # (B,3,d)
        c3 = ctx.unsqueeze(1).expand(-1, 3, -1)
        p3 = p.unsqueeze(1).expand(-1, 3, -1)
        z = self.fuse(torch.cat([c3, r, c3 * r, s_ped.unsqueeze(1).expand(-1, 3, -1), p3], -1))   # (B,3,d)
        rmask = b["rmask"] > 0
        man_logit = self.man(z).squeeze(-1).masked_fill(~rmask, -1e4)
        # trajectories along each route
        q = z.unsqueeze(2) + self.mode_emb.view(1, 1, self.M, -1)                          # (B,3,M,d)
        o = self.traj(q).view(B, 3, self.M, T, 4)
        speed = F.softplus(o[..., 0]) * 3.0
        s_along = torch.cumsum(speed * DT_FUT, -1)                                          # (B,3,M,T) m
        lat = 3.0 * torch.tanh(o[..., 1] / 3.0)
        xy, normal = self.along(b["path"], s_along)
        loc = xy + lat.unsqueeze(-1) * normal
        scale = F.softplus(o[..., 2:]) + 0.05
        mode_logit = self.mode_logit(q).squeeze(-1)                                         # (B,3,M)
        # arrival and conflict per route, occupancy per half
        hz_in = torch.cat([z, s_ped.unsqueeze(1).expand(-1, 3, -1), p3], -1)
        haz = self.haz(hz_in).view(B, 3, 2, NB)
        conf_logit = self.conf(hz_in).squeeze(-1)                                           # (B,3)
        occ_logit = self.occ(torch.cat([p, s_ped, ctx], -1)).view(B, 2, NB)
        return {"man_logit": man_logit, "loc": loc, "scale": scale, "mode_logit": mode_logit,
                "entry_logit": haz[:, :, 0], "exit_logit": haz[:, :, 1], "conf_logit": conf_logit,
                "occ_logit": occ_logit}

    @staticmethod
    def along(path, s):
        """Points at arc length s on uniformly spaced paths (linear extrapolation past the end).
        path (B,3,NP,2), s (B,3,M,T) -> xy (B,3,M,T,2), unit normals (B,3,M,T,2)."""
        B, R, M, Tn = s.shape
        u = s / PATH_SPACING
        idx = torch.clamp(torch.floor(u), 0, NP - 2).long()
        frac = (u - idx.float()).unsqueeze(-1)
        P = path.unsqueeze(2).expand(B, R, M, NP, 2)
        p0 = torch.gather(P, 3, idx.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        p1 = torch.gather(P, 3, (idx + 1).unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        seg = p1 - p0
        xy = p0 + frac * seg
        tng = seg / (seg.norm(dim=-1, keepdim=True) + 1e-6)
        normal = torch.stack([-tng[..., 1], tng[..., 0]], -1)
        return xy, normal


def joint_mode_logprob(out, rmask):
    """log P(route) + log P(mode | route), (B, 3*M); illegal routes get -inf-like values."""
    lp_r = torch.log_softmax(out["man_logit"], -1)                                       # (B,3)
    lp_m = torch.log_softmax(out["mode_logit"], -1)                                      # (B,3,M)
    lp = lp_r.unsqueeze(-1) + lp_m
    lp = lp.masked_fill(~(rmask > 0).unsqueeze(-1), -1e4)
    return lp.flatten(1)


def hazard_pmf(logit):
    h = torch.sigmoid(logit)
    surv = torch.cumprod(1 - h, -1)
    prev = torch.cat([torch.ones_like(surv[..., :1]), surv[..., :-1]], -1)
    return prev * h, surv[..., -1]


@torch.no_grad()
def predict(model, b):
    """Inference summary per sample: modes, route probabilities, entry pmf, occupancy, conflict."""
    out = model(b)
    p_route = torch.softmax(out["man_logit"], -1) * (b["rmask"] > 0)
    p_route = p_route / p_route.sum(-1, keepdim=True).clamp(min=1e-9)
    lp = joint_mode_logprob(out, b["rmask"])
    B = lp.shape[0]
    loc = out["loc"].flatten(1, 2)                                                       # (B,3M,T,2)
    pmf_r, surv_r = hazard_pmf(out["entry_logit"])                                       # (B,3,NB)
    cross = b["crosses"]
    pmf = (p_route.unsqueeze(-1) * pmf_r * cross.unsqueeze(-1)).sum(1)
    p_enter = pmf.sum(-1)
    conf_r = torch.sigmoid(out["conf_logit"]) * cross
    conflict = (p_route * conf_r).sum(-1)
    occ = torch.sigmoid(out["occ_logit"])
    return {"loc": loc, "logp": lp, "p_route": p_route, "entry_pmf": pmf, "p_enter": p_enter,
            "pmf_r": pmf_r, "conflict": conflict, "conf_r": conf_r, "occ": occ, "exit_logit": out["exit_logit"]}
