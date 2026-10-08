"""Noisy SR x4 (sigma = 10/255) with a Gaussian prior on eps (development only; the professor's suggestion).

- Measurement: y = A_sr4(HR) + SIGMA n, SIGMA = 2 * 10/255 (images are in [-1, 1]); one fixed draw per image,
  n = normal(fold_in(PRNGKey(NOISE_SEED), image_index), (1, 64, 64, 3)), stored once in NOISY_Y_NPZ and read back
  by every method.
- CSGM, lambda: minimize ||y - A G(eps)||^2 + lambda ||eps||^2 (Adam lr 0.02, 5,000 steps, restarts from
  initialize(image, r)); keep the restart with the lowest final DATA term. The MAP weight is lambda = SIGMA^2.
  For lambda = 0 the step is the frozen CSGM step itself.
- Regularized BP: x + A^T (A A^T + eta I)^{-1} (y - A x) (iadrift.operators.backproject with eta > 0).
"""

import io
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import paths
from .checkpoint import atomic_write_bytes
from .csgm import TRACE_EVERY
from .metrics import psnr

SIGMA = 2 * 10 / 255                     # 10/255 on the [0, 255] scale, in [-1, 1] units
NOISE_SEED = 2030
NOISE_FLOOR = 64 * 64 * 3 * SIGMA ** 2   # E||sigma n||^2, about 75.6
NOISY_Y_NPZ = "imageNet_x4_noisy_s10.npz"
PSNR_TRACE_EVERY = 250                   # evaluation only (HR), never used for a choice


def noise_draw(image_index):
    key = jax.random.fold_in(jax.random.PRNGKey(NOISE_SEED), int(image_index))
    return np.asarray(jax.random.normal(key, (1, 64, 64, 3), dtype=jnp.float32))[0]


def noisy_measurements(indices, lr_all, data_dir=None):
    """{image_index: noisy y [64, 64, 3]}. Created once in NOISY_Y_NPZ; later calls read it back and check that the
    stored values equal a fresh draw (so every method sees the same y)."""
    path = (data_dir or paths.DATA_DIR) / NOISY_Y_NPZ
    indices = sorted({int(i) for i in indices})
    fresh = {i: (np.asarray(lr_all[i], dtype=np.float32) + np.float32(SIGMA) * noise_draw(i)).astype(np.float32)
             for i in indices}
    if path.exists():
        with np.load(path) as data:
            stored = dict(zip(data["indices"].tolist(), data["y_noisy"]))
            assert float(data["sigma"]) == SIGMA and int(data["seed"]) == NOISE_SEED
        missing = [i for i in indices if i not in stored]
        if missing:
            raise RuntimeError(f"{path} lacks images {missing[:5]}...; extend it deliberately.")
        for i in indices:
            if not np.array_equal(stored[i], fresh[i]):
                raise RuntimeError(f"Stored noisy y of image {i} differs from a fresh draw.")
        return {i: stored[i] for i in indices}
    buffer = io.BytesIO()
    np.savez(buffer, indices=np.asarray(indices, np.int32), y_noisy=np.stack([fresh[i] for i in indices]),
             sigma=SIGMA, seed=NOISE_SEED)
    atomic_write_bytes(path, buffer.getvalue())
    print(f"Saved noisy measurements for {len(indices)} images: {path}")
    return fresh


class NoisyCSGM:
    """CSGM with the penalty lambda ||eps||^2, built on a frozen iadrift.csgm.CSGM instance."""

    def __init__(self, csgm):
        self.csgm = csgm
        measurement_loss, optimizer = csgm.measurement_loss, csgm.optimizer

        def _reg_loss(eps, y, label, noise_labels, lam):
            data = measurement_loss(eps, y, label, noise_labels)
            return data + lam * jnp.sum(eps ** 2), data

        @jax.jit
        def reg_train_step(eps, opt_state, y, label, noise_labels, lam):
            (loss, data), grad = jax.value_and_grad(_reg_loss, has_aux=True)(eps, y, label, noise_labels, lam)
            updates, opt_state = optimizer.update(grad, opt_state, eps)
            return optax.apply_updates(eps, updates), opt_state, loss, data

        self.reg_train_step = reg_train_step

    def reconstruct(self, image_index, restart, label, y_lr, hr, lam, num_steps=None):
        """One CSGM run on (noisy) y_lr [64, 64, 3] with penalty lam ||eps||^2. Returns (metrics, arrays)."""
        c = self.csgm
        num_steps = num_steps or c.num_steps
        y = jnp.asarray(np.asarray(y_lr, np.float32)[None])
        eps, noise_labels = c.initialize(image_index, restart)
        opt_state = c.optimizer.init(eps)
        initial_data = float(c.measurement_loss(eps, y, label, noise_labels))

        trace_steps, trace_loss, trace_data, psnr_steps, psnr_trace = [], [], [], [], []
        t0 = perf_counter()
        for step in range(1, num_steps + 1):
            if lam == 0:
                eps, opt_state, loss_before = c.train_step(eps, opt_state, y, label, noise_labels)
                data_before = loss_before
            else:
                eps, opt_state, loss_before, data_before = self.reg_train_step(
                    eps, opt_state, y, label, noise_labels, jnp.float32(lam))
            if step % TRACE_EVERY == 0:
                trace_steps.append(step - 1)  # evaluated before update `step`
                trace_loss.append(float(loss_before))
                trace_data.append(float(data_before))
            if step % PSNR_TRACE_EVERY == 0:
                psnr_steps.append(step)
                psnr_trace.append(psnr(hr, np.asarray(c.gen.generate(eps, label, noise_labels), np.float32)[0]))
        eps.block_until_ready()
        elapsed = perf_counter() - t0

        final_data = float(c.measurement_loss(eps, y, label, noise_labels))
        x_hat = c.output(eps, label, noise_labels)
        eps_sq = float(jnp.sum(eps ** 2))
        metrics = {
            "lam": float(lam), "initial_data": initial_data, "final_data": final_data,
            "final_penalty": float(lam) * eps_sq, "final_eps_rms": float(np.sqrt(eps_sq / eps.size)),
            "noise_floor": NOISE_FLOOR, "psnr": psnr(hr, x_hat), "seconds": elapsed,
        }
        arrays = {
            "eps": np.asarray(eps, np.float32), "noise_labels": np.asarray(noise_labels), "x_hat": x_hat,
            "trace_steps": np.asarray(trace_steps, np.int32), "trace_loss": np.asarray(trace_loss, np.float32),
            "trace_data": np.asarray(trace_data, np.float32),
            "psnr_steps": np.asarray(psnr_steps, np.int32), "psnr_trace": np.asarray(psnr_trace, np.float32),
        }
        return metrics, arrays
