"""CSGM (Bora et al., 2017) with the drifting generator: keep the weights fixed and optimize the noise input.

    eps_hat = argmin_eps ||y - A G_theta0(eps; z, c)||^2,     x_hat = G_theta0(eps_hat; z, c)

z (the 32 discrete style labels) has no gradient, so it is drawn together with eps at initialization and kept fixed.
Test settings (docs/PROTOCOL.md, Section 3): Adam lr 0.02, 5,000 steps, 2 restarts, no regularizer; of the two restarts
we keep the one with the lower final LR loss.

CSGM.reconstruct is the loop used for the test runs and the ensemble members. CSGM.run is the same loop with any loss
(e.g. A = identity) and an optional projection of eps after each step, used in development experiments.
"""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .metrics import psnr
from .operators import A_sr4

FROZEN_CONFIG = {
    "dev_config_id": 4,       # best setting of the development sweep (scripts/dev/run_csgm_settings_sweep.py)
    "learning_rate": 0.02,
    "num_steps": 5_000,       # the sweep used 3,000; raised to 5,000 before any test run
    "num_restarts": 2,
    "reg_type": "none",
    "reg_weight": 0.0,
}
TRACE_EVERY = 50


def project_none(eps):
    return eps


def project_global_sphere(eps):
    """Rescale eps to the norm of a typical Gaussian draw, sqrt(dim)."""
    return eps * jnp.sqrt(float(eps.size)) / jnp.linalg.norm(eps)


def project_patch_sphere(eps):
    """Rescale each 16x16x3 patch of eps (one generator token) to norm sqrt(768), as in PULSE."""
    b = eps.reshape(1, 16, 16, 16, 16, 3)  # (batch, grid row, pixel row, grid col, pixel col, channel)
    return (b * jnp.sqrt(768.0) / jnp.sqrt(jnp.sum(b ** 2, axis=(2, 4, 5), keepdims=True))).reshape(eps.shape)


class CSGM:
    def __init__(self, gen, learning_rate=FROZEN_CONFIG["learning_rate"], num_steps=FROZEN_CONFIG["num_steps"]):
        self.gen, self.num_steps, self.learning_rate = gen, num_steps, learning_rate
        generate = gen.generate

        def measurement_loss(eps, y, label, noise_labels):
            """||y - A G(eps)||^2, summed over the 64x64x3 measurement."""
            x_hat = generate(eps, label, noise_labels)
            return jnp.sum((A_sr4(x_hat.astype(jnp.float32)) - y) ** 2)

        optimizer = optax.adam(learning_rate=learning_rate)

        @jax.jit
        def train_step(eps, opt_state, y, label, noise_labels):
            loss, grad = jax.value_and_grad(measurement_loss)(eps, y, label, noise_labels)
            updates, opt_state = optimizer.update(grad, opt_state, eps)
            return optax.apply_updates(eps, updates), opt_state, loss

        self.measurement_loss, self.optimizer, self.train_step = measurement_loss, optimizer, train_step
        self._compiled = {}

    def initialize(self, image_index, restart):
        return self.gen.initialize(image_index, restart)

    def output(self, eps, label, noise_labels):
        """G(eps) without jit, float32 [256, 256, 3]."""
        return np.asarray(jax.device_get(self.gen.generate(eps, label, noise_labels)), dtype=np.float32)[0]

    # ------------------------------------------------------------ test loop
    def reconstruct(self, image_index, restart, label, y_lr, hr, num_steps=None, start=None):
        """One CSGM run on y_lr [64, 64, 3]. start: optional (eps, noise_labels) to use instead of initialize().

        Returns (metrics, arrays); arrays holds eps, noise_labels, x_hat and the loss trace (every TRACE_EVERY steps,
        measured before that step's update).
        """
        num_steps = num_steps or self.num_steps
        y = jnp.asarray(np.asarray(y_lr)[None], dtype=jnp.float32)
        eps, noise_labels = start if start is not None else self.initialize(image_index, restart)
        opt_state = self.optimizer.init(eps)
        initial_loss = float(self.measurement_loss(eps, y, label, noise_labels))

        trace_steps, trace_loss = [], []
        t0 = perf_counter()
        for step in range(1, num_steps + 1):
            eps, opt_state, loss_before = self.train_step(eps, opt_state, y, label, noise_labels)
            if step % TRACE_EVERY == 0:
                trace_steps.append(step - 1)  # the step returns the loss before its update
                trace_loss.append(float(loss_before))
        eps.block_until_ready()
        elapsed = perf_counter() - t0

        final_loss = float(self.measurement_loss(eps, y, label, noise_labels))
        x_hat = self.output(eps, label, noise_labels)
        metrics = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "psnr": psnr(hr, x_hat),
            "psnr_clipped": psnr(hr, np.clip(x_hat, -1.0, 1.0)),
            "final_eps_rms": float(jnp.sqrt(jnp.mean(eps ** 2))),
            "final_eps_mean": float(jnp.mean(eps)),
            "final_eps_std": float(jnp.std(eps)),
            "elapsed_seconds": elapsed,
        }
        arrays = {
            "eps": np.asarray(eps, dtype=np.float32),
            "noise_labels": np.asarray(noise_labels),
            "x_hat": x_hat,
            "trace_steps": np.asarray(trace_steps, dtype=np.int32),
            "trace_loss": np.asarray(trace_loss, dtype=np.float32),
        }
        return metrics, arrays

    # ------------------------------------------------------------ development loop
    def step_fn(self, loss_fn, project):
        """Jitted Adam step on loss_fn(eps, target, label, noise_labels) followed by project; cached per pair."""
        if (loss_fn, project) not in self._compiled:
            opt = optax.adam(self.learning_rate)

            @jax.jit
            def step(eps, state, target, label, noise_labels):
                loss, g = jax.value_and_grad(loss_fn)(eps, target, label, noise_labels)
                updates, state = opt.update(g, state, eps)
                return project(optax.apply_updates(eps, updates)), state, loss

            self._compiled[(loss_fn, project)] = (opt, step)
        return self._compiled[(loss_fn, project)]

    def run(self, loss_fn, project, target, label, eps0, noise_labels, num_steps=None):
        """CSGM run with any loss and projection. Returns (metrics, arrays) as in reconstruct."""
        num_steps = num_steps or self.num_steps
        opt, step = self.step_fn(loss_fn, project)
        eps = project(eps0)
        state = opt.init(eps)
        trace_steps, trace_loss = [], []
        t0 = perf_counter()
        for s in range(1, num_steps + 1):
            eps, state, loss_before = step(eps, state, target, label, noise_labels)
            if s % TRACE_EVERY == 0:
                trace_steps.append(s - 1)
                trace_loss.append(float(loss_before))
        eps.block_until_ready()
        metrics = {"final_loss": float(loss_fn(eps, target, label, noise_labels)), "seconds": perf_counter() - t0}
        arrays = {"eps": np.asarray(eps, dtype=np.float32), "noise_labels": np.asarray(noise_labels),
                  "x_hat": self.output(eps, label, noise_labels),
                  "trace_steps": np.asarray(trace_steps, dtype=np.int32),
                  "trace_loss": np.asarray(trace_loss, dtype=np.float32)}
        return metrics, arrays


def best_restart(runs):
    """runs: {restart: (metrics, arrays)} -> (restart, metrics, arrays) of the run with the lowest final LR loss."""
    r = min(runs, key=lambda r: (runs[r][0]["final_loss"], r))
    return (r, *runs[r])
