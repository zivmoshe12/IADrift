"""Posterior-mean ensembles of CSGM reconstructions (docs/PROTOCOL.md §11, Addendum A).

Members (correct label): the 2 CSGM restarts r0, r1, plus 3 CSGM runs on the measurement circularly shifted by
s in {(0,1), (1,0), (1,1)} LR pixels from the r0 start, shifted back by 4s HR pixels:

    x_s = S_{-4s} CSGM(S_s y)

Ensembles (all followed by back-projection):
    E1 = mean of r0, r1;   E2 = mean of r0 + 3 shifts;   E3 = mean of all 5;
    E3w = weighted mean of all 5, w_k ~ exp(-(L_k - L_min) / std(L)), L_k = ||y - A x_k||^2 (never HR)  [primary]
"""

import numpy as np

from .operators import backproject

SHIFTS = ((0, 1), (1, 0), (1, 1))
START_RESTART = 0  # every shifted member starts from initialize(image, 0), like restart r0
MEMBERS = ("r0", "r1", "shift01", "shift10", "shift11")
ENSEMBLES = {
    "E1: mean of 2 restarts": (("r0", "r1"), False),
    "E2: mean of r0 + 3 shifts": (("r0", "shift01", "shift10", "shift11"), False),
    "E3: mean of 5 CSGM runs": (MEMBERS, False),
    "E3w: weighted 5 CSGM runs": (MEMBERS, True),
}


def shift_y(y_lr, sy, sx):
    """[64, 64, 3] -> circularly shifted copy (= jnp.roll(y[None], (sy, sx), axis=(1, 2))[0])."""
    return np.roll(np.asarray(y_lr, dtype=np.float32), (sy, sx), axis=(0, 1))


def unshift_x(x_hat, sy, sx):
    """Shift a reconstruction of the shifted measurement back to the original frame (4 HR pixels per LR pixel)."""
    return np.roll(x_hat, (-4 * sy, -4 * sx), axis=(0, 1))


def weights(lr_losses, weighted):
    """Uniform, or w_k ~ exp(-(L_k - L_min) / T) with T = std(L) (an untuned heuristic; never uses HR)."""
    L = np.asarray(lr_losses, np.float64)
    w = np.exp(-(L - L.min()) / max(L.std(), 1e-6)) if weighted else np.ones(len(L))
    return w / w.sum()


def ensemble(members, lr_losses, y, weighted):
    """members: list of [256, 256, 3]; lr_losses: their ||y - A x_k||^2. Returns (BP of the (weighted) mean, weights)."""
    w = weights(lr_losses, weighted)
    return backproject(sum(wk * x for wk, x in zip(w, members)), y), w


def effective_members(w):
    """1 / sum w_k^2: how many members the weighted mean effectively uses."""
    return float(1.0 / np.sum(np.asarray(w) ** 2))
