"""Training objectives for SA-V2X-Ego."""
import torch
import torch.nn.functional as F

from sav2x.features import NB
from sav2x.model import joint_mode_logprob

W = {"traj": 1.0, "man": 0.5, "entry": 0.3, "exit": 0.2, "occ": 0.3, "conf": 1.0}


def traj_loss(out, b):
    """Route-conditioned winner-takes-all.

    When the true maneuver is known, the winner is chosen among the true route's modes only and
    the classification term is CE over P(mode | route); P(route) is trained by the maneuver loss
    alone. Otherwise (rare), the winner is chosen among all legal modes with CE on the joint
    probability. Conditioning matters for stopped vehicles: every route then predicts nearly the
    same 8 s trajectory, and an unconditioned winner would push P(route) toward an arbitrary route."""
    B, R, M = out["mode_logit"].shape
    loc = out["loc"].flatten(1, 2)                                   # (B,R*M,T,2)
    scale = out["scale"].flatten(1, 2)
    tgt, m = b["fut"], b["fmask"].float()                            # (B,T,2), (B,T)
    err = torch.linalg.norm(loc - tgt.unsqueeze(1), dim=-1)          # (B,K,T)
    n = m.sum(-1).clamp(min=1)
    ade = (err * m.unsqueeze(1)).sum(-1) / n.unsqueeze(1)            # (B,K)
    y = b["man"]
    known = y >= 0
    route_of_mode = torch.arange(R, device=ade.device).repeat_interleave(M)          # (K,)
    legal = (b["rmask"] > 0).repeat_interleave(M, dim=1)
    allowed = torch.where(known.unsqueeze(1), route_of_mode.unsqueeze(0) == y.clamp(min=0).unsqueeze(1), legal)
    best = ade.masked_fill(~allowed, 1e6).argmin(-1)
    g = best.view(B, 1, 1, 1).expand(-1, 1, loc.shape[2], 2)
    lb, sb = loc.gather(1, g).squeeze(1), scale.gather(1, g).squeeze(1)
    nll = ((torch.abs(lb - tgt) / sb + torch.log(2 * sb)).sum(-1) * m).sum(-1) / n
    lp_joint = joint_mode_logprob(out, b["rmask"])                    # (B,K)
    lp_cond = torch.log_softmax(out["mode_logit"], -1).flatten(1)     # (B,K): log P(mode | its route)
    lp = torch.where(known.unsqueeze(1), lp_cond, lp_joint)
    ce = -lp.gather(1, best.unsqueeze(1)).squeeze(1)
    ok = (m.sum(-1) > 0).float()
    ade_legal = ade.masked_fill(~legal, 1e6).min(-1).values
    return ((nll + ce) * ok).sum() / ok.sum().clamp(min=1), (ade_legal * ok).sum() / ok.sum().clamp(min=1)


def man_loss(out, b):
    y = b["man"]
    ok = y >= 0
    if not ok.any():
        return out["man_logit"].sum() * 0
    return F.cross_entropy(out["man_logit"][ok], y[ok])


def _surv(logit):
    log_h, log_s = F.logsigmoid(logit), F.logsigmoid(-logit)
    cum0 = torch.cat([torch.zeros_like(log_s[..., :1]), torch.cumsum(log_s, -1)], -1)
    return log_h, cum0


def arrival_loss(out, b):
    """Censoring-aware discrete-time survival on the true route, for routes that cross the target."""
    y = b["man"]
    e_bin, e_evt, x_bin, x_evt = b["surv"].unbind(-1)
    ok = (y >= 0) & (b["crosses"].gather(1, y.clamp(min=0).unsqueeze(1)).squeeze(1) > 0) & ~b["inside"]
    if not ok.any():
        z = out["entry_logit"].sum() * 0
        return z, z
    idx = y.clamp(min=0).view(-1, 1, 1).expand(-1, 1, NB)
    el = out["entry_logit"].gather(1, idx).squeeze(1)[ok]
    xl = out["exit_logit"].gather(1, idx).squeeze(1)[ok]
    eb, ee, xb, xe = e_bin[ok], e_evt[ok], x_bin[ok], x_evt[ok]
    log_h, cum0 = _surv(el)
    nll_e = -cum0.gather(1, eb.clamp(0, NB).unsqueeze(1)).squeeze(1)
    nll_e = nll_e - torch.where(ee == 1, log_h.gather(1, eb.clamp(0, NB - 1).unsqueeze(1)).squeeze(1),
                                torch.zeros_like(nll_e))
    log_h2, cum2 = _surv(xl)
    has = (xe >= 0) & (ee == 1)
    ebc = eb.clamp(0, NB - 1)
    xbc = torch.maximum(xb, ebc).clamp(0, NB)
    nll_x = -(cum2.gather(1, xbc.unsqueeze(1)) - cum2.gather(1, ebc.unsqueeze(1))).squeeze(1)
    nll_x = nll_x - torch.where(xe == 1, log_h2.gather(1, xbc.clamp(0, NB - 1).unsqueeze(1)).squeeze(1),
                                torch.zeros_like(nll_x))
    lx = (nll_x * has.float()).sum() / has.float().sum().clamp(min=1)
    return nll_e.mean(), lx


def occ_loss(out, b):
    m = b["omask"].unsqueeze(1).expand_as(b["occ"])
    l = F.binary_cross_entropy_with_logits(out["occ_logit"], b["occ"], reduction="none")
    return (l * m).sum() / m.sum().clamp(min=1)


def conf_loss(out, b, pos_weight=2.0):
    """Conflict head on the true route; routes that do not cross the target are zero by construction."""
    y = b["man"]
    ok = (y >= 0) & (b["conf_known"] > 0) & (b["crosses"].gather(1, y.clamp(min=0).unsqueeze(1)).squeeze(1) > 0)
    if not ok.any():
        return out["conf_logit"].sum() * 0
    lg = out["conf_logit"].gather(1, y.clamp(min=0).unsqueeze(1)).squeeze(1)[ok]
    t = b["conf"][ok]
    return F.binary_cross_entropy_with_logits(lg, t, pos_weight=torch.tensor(pos_weight, device=lg.device))


def total_loss(out, b):
    lt, ade = traj_loss(out, b)
    lm = man_loss(out, b)
    le, lx = arrival_loss(out, b)
    lo = occ_loss(out, b)
    lc = conf_loss(out, b)
    loss = W["traj"] * lt + W["man"] * lm + W["entry"] * le + W["exit"] * lx + W["occ"] * lo + W["conf"] * lc
    return loss, {"traj": lt, "ade": ade, "man": lm, "entry": le, "exit": lx, "occ": lo, "conf": lc}
