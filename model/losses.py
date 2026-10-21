"""Training objectives: winner-takes-all Laplace trajectories, maneuver CE, and
censoring-aware discrete-time survival losses for crosswalk entry / exit."""
import torch
import torch.nn.functional as F

from model.net import to_agent_frame


def traj_loss(out, b):
    tgt = to_agent_frame(b["fut"], b["pose"])                       # (B,A,T,2)
    m = b["fmask"].float()                                          # (B,A,T)
    loc, scale = out["loc"], out["scale"]                           # (B,A,K,T,2)
    err = torch.linalg.norm(loc - tgt.unsqueeze(2), dim=-1)         # (B,A,K,T)
    n = m.sum(-1).clamp(min=1)
    ade = (err * m.unsqueeze(2)).sum(-1) / n.unsqueeze(2)           # (B,A,K)
    best = ade.argmin(-1)                                           # (B,A)
    g = best.view(*best.shape, 1, 1, 1).expand(-1, -1, 1, loc.shape[3], 2)
    lb = loc.gather(2, g).squeeze(2)
    sb = scale.gather(2, g).squeeze(2)
    nll = (torch.abs(lb - tgt) / sb + torch.log(2 * sb)).sum(-1)    # (B,A,T)
    nll = (nll * m).sum(-1) / n
    ce = F.cross_entropy(out["mode_logit"].flatten(0, 1), best.flatten(), reduction="none").view(best.shape)
    w = b["weight"] * b["amask"].float() * (m.sum(-1) > 0).float()
    return ((nll + ce) * w).sum() / w.sum().clamp(min=1), (ade.min(-1).values * w).sum() / w.sum().clamp(min=1)


def maneuver_loss(out, b):
    y = b["man"]
    ok = (y >= 0) & b["amask"]
    if not ok.any():
        return out["man_logit"].sum() * 0
    logit = out["man_logit"][ok]
    yy = y[ok]
    legal = out["man_ok"][ok].gather(1, yy.view(-1, 1)).squeeze(1)
    if not legal.any():
        return out["man_logit"].sum() * 0
    w = b["weight"][ok][legal]
    ce = F.cross_entropy(logit[legal], yy[legal], reduction="none")
    return (ce * w).sum() / w.sum().clamp(min=1)


def _surv_terms(logit):
    log_h = F.logsigmoid(logit)
    log_s = F.logsigmoid(-logit)
    cum = torch.cumsum(log_s, -1)
    cum0 = torch.cat([torch.zeros_like(cum[..., :1]), cum], -1)     # cum0[k] = sum_{j<k} log(1-h_j)
    return log_h, cum0


def hazard_loss(out, b, nb):
    e_bin, e_evt, x_bin, x_evt = b["surv"].unbind(-1)
    w = b["weight"] * b["amask"].float()
    # entry
    log_h, cum0 = _surv_terms(out["entry_logit"])
    kb = e_bin.clamp(0, nb)
    nll_e = -cum0.gather(-1, kb.unsqueeze(-1)).squeeze(-1)
    ev = e_evt == 1
    nll_e = nll_e - torch.where(ev, log_h.gather(-1, e_bin.clamp(0, nb - 1).unsqueeze(-1)).squeeze(-1),
                                torch.zeros_like(nll_e))
    le = (nll_e * w).sum() / w.sum().clamp(min=1)
    # exit, conditional on entry bin
    log_h2, cum2 = _surv_terms(out["exit_logit"])
    has = (x_evt >= 0) & ev
    eb = e_bin.clamp(0, nb - 1)
    xb = torch.maximum(x_bin, eb).clamp(0, nb)
    nll_x = -(cum2.gather(-1, xb.unsqueeze(-1)) - cum2.gather(-1, eb.unsqueeze(-1))).squeeze(-1)
    nll_x = nll_x - torch.where(x_evt == 1, log_h2.gather(-1, xb.clamp(0, nb - 1).unsqueeze(-1)).squeeze(-1),
                                torch.zeros_like(nll_x))
    wx = w * has.float()
    lx = (nll_x * wx).sum() / wx.sum().clamp(min=1)
    return le, lx
