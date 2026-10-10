"""The SR x4 measurement operator and the operations built on it.

    A_sr4(x)           antialiased bicubic downsampling 256 -> 64, a fixed linear map: y = A x*
    bicubic_up(y)      bicubic baseline 64 -> 256
    backproject(x, y)  BP(x) = x + A^T (A A^T)^{-1} (y - A x), the orthogonal projection onto {x' : A x' = y}
    lr_loss(x, y)      ||y - A x||^2

A^T comes from jax.linear_transpose; (A A^T) u = r is solved by conjugate gradients (tol 1e-6, at most 500 iterations).
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

HR_SHAPE = (256, 256, 3)
LR_SHAPE = (64, 64, 3)
LR_NYQUIST = 0.125  # highest frequency a 64x64 measurement holds, in cycles per HR pixel


def A_sr4(x):
    """Antialiased bicubic: [batch, 256, 256, 3] -> [batch, 64, 64, 3]."""
    assert x.ndim == 4 and x.shape[1:] == HR_SHAPE
    return jax.image.resize(x.astype(jnp.float32), (x.shape[0], 64, 64, 3), method="cubic", antialias=True)


def bicubic_up(y):
    """[64, 64, 3] -> [256, 256, 3] bicubic upsampling."""
    return np.asarray(jax.image.resize(jnp.asarray(y, dtype=jnp.float32), HR_SHAPE, method="cubic", antialias=False))


_A_transpose = jax.linear_transpose(A_sr4, jnp.zeros((1,) + HR_SHAPE, jnp.float32))


def A_transpose(v):
    """A^T: [1, 64, 64, 3] -> [1, 256, 256, 3]."""
    return _A_transpose(v)[0]


@jax.jit
def _backproject(x, y):
    u, _ = cg(lambda v: A_sr4(A_transpose(v)), y - A_sr4(x), tol=1e-6, maxiter=500)  # (A A^T) u = y - A x
    return x + A_transpose(u)


def backproject(x, y):
    """Closest image to x [256, 256, 3] that reproduces y [64, 64, 3] exactly."""
    x = jnp.asarray(x, jnp.float32)[None]
    y = jnp.asarray(np.asarray(y, np.float32))[None]
    out = _backproject(x, y)
    return np.asarray(out[0])


def lr_loss(x, y):
    """||y - A x||^2 for x [256, 256, 3], y [64, 64, 3]."""
    y = jnp.asarray(np.asarray(y, np.float32))[None]
    return float(jnp.sum((A_sr4(jnp.asarray(x, jnp.float32)[None]) - y) ** 2))

