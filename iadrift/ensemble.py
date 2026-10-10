"""Posterior-mean ensembles of CSGM reconstructions (docs/PROTOCOL.md, Section 11).

Five members per image: the two CSGM restarts r0, r1 and three CSGM runs on the measurement circularly shifted by
s in {(0,1), (1,0), (1,1)} LR pixels, started from the r0 initialization and shifted back by 4s HR pixels:

    x_s = S_{-4s} CSGM(S_s y)

Each ensemble is a (weighted) mean of members followed by back-projection:

    E1     mean of r0, r1
    E2     mean of r0 and the 3 shifts
    Ens-U  mean of all 5
    Ens-W  weighted mean of all 5, w_k ~ exp(-(L_k - L_min) / std(L)) with L_k = ||y - A x_k||^2  (primary)
"""

import numpy as np

from .operators import backproject

SHIFTS = ((0, 1), (1, 0), (1, 1))
START_RESTART = 0  # shifted members start from initialize(image, 0), the r0 start
MEMBERS = ("r0", "r1", "shift01", "shift10", "shift11")
ENSEMBLES = {
    "E1: mean of 2 restarts": (("r0", "r1"), False),
    "E2: mean of r0 + 3 shifts": (("r0", "shift01", "shift10", "shift11"), False),
    "Ens-U: mean of 5 CSGM runs": (MEMBERS, False),
    "Ens-W: weighted 5 CSGM runs": (MEMBERS, True),
}


def shift_y(y_lr, sy, sx):
    """Circular shift of a [64, 64, 3] measurement by (sy, sx) LR pixels."""
    return np.roll(np.asarray(y_lr, dtype=np.float32), (sy, sx), axis=(0, 1))


def unshift_x(x_hat, sy, sx):
    """Undo shift_y on a [256, 256, 3] reconstruction (4 HR pixels per LR pixel)."""
    return np.roll(x_hat, (-4 * sy, -4 * sx), axis=(0, 1))


def weights(lr_losses, weighted):
    """Normalized weights: uniform, or w_k ~ exp(-(L_k - L_min) / std(L)). The temperature std(L) was not tuned."""
    L = np.asarray(lr_losses, np.float64)
    w = np.exp(-(L - L.min()) / max(L.std(), 1e-6)) if weighted else np.ones(len(L))
    return w / w.sum()


def ensemble(members, lr_losses, y, weighted):
    """members: list of [256, 256, 3]; lr_losses: their ||y - A x_k||^2. Returns (back-projected mean, weights)."""
    w = weights(lr_losses, weighted)
    return backproject(sum(wk * x for wk, x in zip(w, members)), y), w


def effective_members(w):
    """Effective number of members, 1 / sum w_k^2."""
    return float(1.0 / np.sum(np.asarray(w) ** 2))
