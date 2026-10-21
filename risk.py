"""Conflict risk and alerting on top of the predictor's crosswalk hazards.

risk(v, p) = P( vehicle occupancy [Tv_in, Tv_out] and pedestrian occupancy
[Tp_in, Tp_out] overlap with margin theta ), estimated by Monte Carlo from each
agent's entry pmf and entry-conditional exit hazards. Vehicle and pedestrian
are treated as independent given the scene (a simplification: yielding couples
them; see the design doc).

The raw probability is then calibrated (isotonic regression on a validation
split) so it can be read as a risk confidence, and an alert fires when

    calibrated risk >= p_star  and  predicted vehicle entry within [0, tta_max]

with severity from the lead time actually available versus the speed-dependent
minimum TTA_min(v) = t_sys + t_PR + v / (2 a_c).
"""
import numpy as np
import torch

A_COMF = 3.4      # AASHTO design deceleration, m/s^2
T_PR = 1.5        # warned-driver perception-reaction time, s (2.5 s = AASHTO design)
T_SYS = 0.3       # end-to-end pipeline latency budget, s


def tta_min(v, t_sys=T_SYS, t_pr=T_PR, a=A_COMF):
    return t_sys + t_pr + np.asarray(v) / (2 * a)


def exit_pmf_given_entry(exit_logit):
    """(N, nb) exit hazard logits -> (N, nb, nb+1) pmf of exit bin given entry bin e (last = beyond)."""
    h = torch.sigmoid(exit_logit)                         # (N,nb)
    nb = h.shape[-1]
    log1m = torch.log1p(-h.clamp(max=1 - 1e-6))
    cum = torch.cat([torch.zeros_like(log1m[:, :1]), torch.cumsum(log1m, -1)], -1)   # (N,nb+1)
    k = torch.arange(nb)
    e = torch.arange(nb)
    # survive bins e..k-1 then exit at k (k >= e)
    surv = torch.exp(cum[:, None, :nb] - cum[:, e, None])                            # (N,e,k)
    pmf = surv * h[:, None, :] * (k[None, :] >= e[:, None]).float()
    beyond = torch.exp(cum[:, None, nb] - cum[:, e])                                 # (N,e)
    return torch.cat([pmf, beyond.unsqueeze(-1)], -1)


def sample_intervals(entry_pmf, entry_surv, exit_cond, bin_s, n=256, gen=None):
    """Sample occupancy intervals. Returns tin, tout (N, n) seconds; never-entering -> inf."""
    N, nb = entry_pmf.shape
    probs = torch.cat([entry_pmf, entry_surv.unsqueeze(-1)], -1).clamp(min=0)
    probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-9)
    e = torch.multinomial(probs, n, replacement=True, generator=gen)                 # (N,n)
    never = e == nb
    ec = e.clamp(max=nb - 1)
    xp = exit_cond[torch.arange(N)[:, None], ec]                                     # (N,n,nb+1)
    x = torch.multinomial(xp.reshape(-1, nb + 1).clamp(min=1e-12), 1, generator=gen).view(N, n)
    u1 = torch.rand(N, n, generator=gen)
    u2 = torch.rand(N, n, generator=gen)
    tin = (ec.float() + u1) * bin_s
    tout = torch.where(x == nb, torch.full_like(tin, nb * bin_s + 5.0), (x.float() + u2) * bin_s)
    tout = torch.maximum(tout, tin + 0.1)
    tin = torch.where(never, torch.full_like(tin, float("inf")), tin)
    tout = torch.where(never, torch.full_like(tout, float("inf")), tout)
    return tin, tout


def pair_risk(v_in, v_out, p_in, p_out, theta=1.0):
    """Overlap probability for every vehicle x pedestrian pair. Inputs (V,n) and (P,n) -> (V,P)."""
    a = v_in[:, None, :] < p_out[None, :, :] + theta
    b = v_out[:, None, :] > p_in[None, :, :] - theta
    fin = torch.isfinite(v_in)[:, None, :] & torch.isfinite(p_in)[None, :, :]
    return (a & b & fin).float().mean(-1)


class Calibrator:
    """Isotonic calibration of raw risk to observed conflict frequency."""

    def __init__(self):
        self.iso = None

    def fit(self, p, y):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def __call__(self, p):
        return p if self.iso is None else self.iso.predict(np.asarray(p))


def expected_calibration_error(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
