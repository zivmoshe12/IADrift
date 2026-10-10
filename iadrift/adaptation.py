"""IADrift: image-adaptive inversion in the style of IAGAN (Hussein et al., 2020) with the drifting generator.

Starting from the CSGM solution (theta0, eps_CSGM), adapt the generator weights and the noise input jointly:

    (theta_hat, eps_hat) = argmin_{theta, eps} ||y - A G_theta(eps; z, c)||^2,     x_hat = G_theta_hat(eps_hat; z, c)

Test settings are frozen in docs/PROTOCOL.md, Section 4: Adam with lr 1e-5 on all 241 weight tensors (134M parameters)
and a separate Adam with lr 1e-3 on eps, 1,000 steps, outputs taken at steps 100 and 1,000.
Theta-only adaptation (eps fixed) is a development diagnostic.

Run with iadrift.runtime.setup(theta_gradients=True); see runtime.py.
"""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .metrics import psnr
from .operators import A_sr4

FROZEN_CONFIG = {
    "lr_theta": 1e-5,
    "lr_eps": 1e-3,
    "num_steps": 1_000,
    "checkpoints": [100, 1_000],
    "cfg": 1.0,
}
TRACE_EVERY = 50


class Adaptation:
    """Loss and jitted update steps for weight adaptation."""

    def __init__(self, gen):
        self.gen = gen
        generate = gen.generate_theta

        def iadrift_loss(theta, eps, y, label, noise_labels):
            return jnp.sum((A_sr4(generate(theta, eps, label, noise_labels)) - y) ** 2)

        self.loss = iadrift_loss
        self.generate_jit = jax.jit(generate)

    def make_joint_step(self, lr_theta, lr_eps):
        """Joint IADrift step. theta, eps and both optimizer states are donated, so pass a copy of the pretrained
        weights (fresh_weights), never gen.params."""
        opt_theta, opt_eps = optax.adam(lr_theta), optax.adam(lr_eps)
        loss_fn = self.loss

        def step(theta, eps, state_theta, state_eps, y, label, noise_labels):
            loss, (g_theta, g_eps) = jax.value_and_grad(loss_fn, argnums=(0, 1))(theta, eps, y, label, noise_labels)
            updates, state_theta = opt_theta.update(g_theta, state_theta, theta)
            theta = optax.apply_updates(theta, updates)
            updates, state_eps = opt_eps.update(g_eps, state_eps, eps)
            eps = optax.apply_updates(eps, updates)
            return theta, eps, state_theta, state_eps, loss

        return opt_theta, opt_eps, jax.jit(step, donate_argnums=(0, 1, 2, 3))

    def make_theta_step(self, lr_theta):
        """Theta-only step with eps fixed. theta and its optimizer state are donated."""
        opt = optax.adam(lr_theta)
        loss_fn = self.loss

        def step(theta, state, eps, y, label, noise_labels):
            loss, g = jax.value_and_grad(loss_fn, argnums=0)(theta, eps, y, label, noise_labels)
            updates, state = opt.update(g, state, theta)
            return optax.apply_updates(theta, updates), state, loss

        return opt, jax.jit(step, donate_argnums=(0, 1))

    def eager_metrics(self, theta, eps, label, noise_labels, y, hr):
        """Un-jitted evaluation, the same path used for CSGM outputs. Returns (x [256, 256, 3], LR loss, PSNR)."""
        x = self.gen.generate_theta(theta, eps, label, noise_labels)
        loss = float(jnp.sum((A_sr4(x) - y) ** 2))
        x = np.asarray(x[0])
        return x, loss, psnr(hr, x)

    def fresh_weights(self):
        """Copy of the pretrained weights."""
        return jax.tree_util.tree_map(jnp.copy, self.gen.params)

    def run_iadrift(self, eps, noise_labels, label, y, hr, joint_step, opt_theta, opt_eps,
                    num_steps=FROZEN_CONFIG["num_steps"], checkpoints=tuple(FROZEN_CONFIG["checkpoints"])):
        """One IADrift run starting from the CSGM solution (eps, z). y: [1, 64, 64, 3].

        Returns (metrics, arrays): output and eps at each checkpoint step, plus the loss trace every TRACE_EVERY steps.
        """
        x0, loss0, psnr0 = self.eager_metrics(self.gen.params, eps, label, noise_labels, y, hr)
        metrics = {"start_lr_loss": loss0, "start_psnr": psnr0}
        theta = self.fresh_weights()
        state_theta, state_eps = opt_theta.init(theta), opt_eps.init(eps)
        arrays, trace_steps, trace_loss = {"x_start": np.asarray(x0, np.float32)}, [], []
        t0 = perf_counter()
        for step in range(1, num_steps + 1):
            theta, eps, state_theta, state_eps, loss_before = joint_step(
                theta, eps, state_theta, state_eps, y, label, noise_labels)
            if step % TRACE_EVERY == 0:
                trace_steps.append(step - 1)  # the step returns the loss before its update
                trace_loss.append(float(loss_before))
            if step in checkpoints:
                x, loss, p = self.eager_metrics(theta, eps, label, noise_labels, y, hr)
                finite = bool(np.all(np.isfinite(x)) and np.isfinite(loss))
                metrics[f"step{step}"] = {"lr_loss": loss, "psnr": p, "psnr_clipped": psnr(hr, np.clip(x, -1, 1)),
                                          "finite": finite}
                arrays[f"x_step{step}"] = np.asarray(x, np.float32)
                arrays[f"eps_step{step}"] = np.asarray(eps, np.float32)
        metrics["elapsed_seconds"] = perf_counter() - t0
        arrays["trace_steps"] = np.asarray(trace_steps, np.int32)
        arrays["trace_loss"] = np.asarray(trace_loss, np.float32)
        arrays["noise_labels"] = np.asarray(noise_labels)
        del theta, state_theta, state_eps
        return metrics, arrays
