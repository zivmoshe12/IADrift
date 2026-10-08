"""The measurement operator of SR x4 and the operations built on it.

    A_sr4(x)           antialiased bicubic downsampling 256 -> 64 (a fixed linear map; y = A x*)
    bicubic_up(y)      the bicubic baseline 64 -> 256
    backproject(x, y)  BP(x) = x + A^T (A A^T + eta I)^{-1} (y - A x); eta = 0 is the orthogonal projection onto
                       {x' : A x' = y} (parameter-free), eta > 0 the regularized version for noisy y
    lr_loss(x, y)      ||y - A x||^2

A^T is obtained by automatic transposition (jax.linear_transpose) and (A A^T + eta I) u = r is solved with conjugate
gradients (tolerance 1e-6, at most 500 iterations).
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

HR_SHAPE = (256, 256, 3)
LR_SHAPE = (64, 64, 3)
LR_NYQUIST = 0.125  # cycles per HR pixel: the highest frequency a 64x64 measurement holds


def A_sr4(x):
    """Antialiased bicubic: [batch, 256, 256, 3] -> [batch, 64, 64, 3]."""
    assert x.ndim == 4 and x.shape[1:] == HR_SHAPE
    return jax.image.resize(x.astype(jnp.float32), (x.shape[0], 64, 64, 3), method="cubic", antialias=True)


def bicubic_up(y):
    """[64, 64, 3] -> [256, 256, 3], the bicubic baseline (no antialiasing needed when upsampling)."""
    return np.asarray(jax.image.resize(jnp.asarray(y, dtype=jnp.float32), HR_SHAPE, method="cubic", antialias=False))


_A_transpose = jax.linear_transpose(A_sr4, jnp.zeros((1,) + HR_SHAPE, jnp.float32))


def A_transpose(v):
    """A^T: [1, 64, 64, 3] -> [1, 256, 256, 3]."""
    return _A_transpose(v)[0]


@jax.jit
def _backproject(x, y):
    u, _ = cg(lambda v: A_sr4(A_transpose(v)), y - A_sr4(x), tol=1e-6, maxiter=500)  # (A A^T) u = y - A x
    return x + A_transpose(u)


@jax.jit
def _backproject_reg(x, y, eta):
    u, _ = cg(lambda v: A_sr4(A_transpose(v)) + eta * v, y - A_sr4(x), tol=1e-6, maxiter=500)
    return x + A_transpose(u)


def backproject(x, y, eta=0.0):
    """x [256, 256, 3], y [64, 64, 3] -> back-projected image [256, 256, 3].

    eta = 0: the closest image to x that reproduces y exactly (the noiseless BP of all main results).
    eta > 0: x + A^T (A A^T + eta I)^{-1} (y - A x), for noisy measurements.
    """
    x = jnp.asarray(x, jnp.float32)[None]
    y = jnp.asarray(np.asarray(y, np.float32))[None]
    out = _backproject(x, y) if eta == 0 else _backproject_reg(x, y, jnp.float32(eta))
    return np.asarray(out[0])


def lr_loss(x, y):
    """||y - A x||^2 for x [256, 256, 3], y [64, 64, 3]."""
    y = jnp.asarray(np.asarray(y, np.float32))[None]
    return float(jnp.sum((A_sr4(jnp.asarray(x, jnp.float32)[None]) - y) ** 2))


def lambda_max_AAt(iters=300, seed=0):
    """Largest eigenvalue of A A^T, by power iteration (scale for the regularized BP's eta)."""
    v = jax.random.normal(jax.random.PRNGKey(seed), (1,) + LR_SHAPE, jnp.float32)
    op = jax.jit(lambda v: A_sr4(A_transpose(v)))
    lam = 0.0
    for _ in range(iters):
        w = op(v / jnp.linalg.norm(v))
        lam, v = float(jnp.linalg.norm(w)), w
    return lam
