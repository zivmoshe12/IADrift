"""The prior: the pretrained drifting model ``pixel_B_sota`` (Deng et al., 2026), a one-pass generator.

    x = G_theta(eps; z, c),  eps in R^{256x256x3} (Gaussian pixel noise), z in {0..63}^32 (discrete noise labels),
                             c in {0..999} (class), classifier-free-guidance scale 1.0, output in [-1, 1].

The calls below are exactly those of the upstream ``models/generator.py`` API used in every experiment:
``c_cfg_noise_to_cond(c, cfg, z)`` -> ``generate_image(eps, cond, deterministic=True)``.
Call ``iadrift.runtime.setup`` before importing this module.
"""

import os

import jax
import jax.numpy as jnp

from . import paths
from .data import SWEEP_SEED

MODEL_ID = "hf://pixel_B_sota"


class Generator:
    """The pretrained generator and the calls used by the methods.

    generate(eps, label, z)              frozen pretrained weights (CSGM). The weights are closed over, exactly as in
                                         the original benchmark script, so the compiled programs (and their bf16
                                         numerics) are the same.
    generate_theta(theta, eps, label, z) weights as an argument (IADrift, theta-only adaptation); returns float32.
    generate_cfg(eps, label, z, cfg)     frozen weights, any guidance scale (random samples).
    """

    def __init__(self, model, params):
        self.model, self.params = model, params
        self.noise_coords, self.noise_classes = model.noise_coords, model.noise_classes

        def generate(eps, label, noise_labels):
            condition = model.apply(
                {"params": params}, jnp.asarray(label, dtype=jnp.int32).reshape(1),
                1.0, noise_labels, method=model.c_cfg_noise_to_cond,
            )
            return model.apply({"params": params}, eps, condition, True, method=model.generate_image)

        def generate_theta(theta, eps, label, noise_labels):
            condition = model.apply(
                {"params": theta}, jnp.asarray(label, dtype=jnp.int32).reshape(1),
                1.0, noise_labels, method=model.c_cfg_noise_to_cond,
            )
            x_hat = model.apply({"params": theta}, eps, condition, True, method=model.generate_image)
            return x_hat.astype(jnp.float32)

        def generate_cfg(eps, label, noise_labels, cfg):
            condition = model.apply(
                {"params": params}, jnp.asarray(label, dtype=jnp.int32).reshape(1),
                cfg, noise_labels, method=model.c_cfg_noise_to_cond,
            )
            return model.apply({"params": params}, eps, condition, True, method=model.generate_image)

        self.generate = generate
        self.generate_theta = generate_theta
        self.generate_cfg = generate_cfg

    def initialize(self, image_index, restart, seed=SWEEP_SEED):
        """Random start (eps, z) for one image and restart: key(seed, image, restart). Independent of the class label,
        so correct- and wrong-label runs of the same image and restart start from the same point."""
        key = jax.random.PRNGKey(seed)
        key = jax.random.fold_in(key, image_index)
        key = jax.random.fold_in(key, restart)
        key_eps, key_style = jax.random.split(key)
        eps = jax.random.normal(key_eps, (1, 256, 256, 3), dtype=jnp.float32)
        noise_labels = jax.random.randint(
            key_style, shape=(1, self.noise_coords), minval=0, maxval=self.noise_classes, dtype=jnp.int32,
        )
        return eps, noise_labels


def load_generator():
    """Load pixel_B_sota (downloaded into IADRIFT_HF_CACHE on first use)."""
    from utils.init_util import load_generator_model_and_params  # upstream drifting code (DRIFTING_DIR)

    model, params, _ = load_generator_model_and_params(MODEL_ID, hf_cache_dir=str(paths.HF_CACHE_DIR))
    if os.environ.get("IADRIFT_REQUIRE_GPU", "1") == "1":
        assert jax.default_backend() == "gpu", "Refusing to run the experiments on CPU."
    return Generator(model, params)
