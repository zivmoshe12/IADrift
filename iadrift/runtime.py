"""Process set-up that must happen before JAX is imported.

Every runner starts with

    from iadrift.runtime import setup
    setup(theta_gradients=False)   # CSGM-type runs
    setup(theta_gradients=True)    # runs with gradients w.r.t. the generator weights (IADrift, theta-only)

Two modes because of reproducibility (README, "Numerics and verification"). The generator computes in bf16 and CSGM
trajectories are chaotic, so a run is bit-exactly reproducible only under the same XLA allocator: CSGM runs match the
saved runs with JAX's default GPU allocator, while gradients w.r.t. all 134M weights need
XLA_PYTHON_CLIENT_ALLOCATOR=platform because the default allocator hangs under WSL2. Keep the two kinds of run in
separate processes.
"""

import os
import sys

from . import paths


def setup(theta_gradients=False, require_gpu=True):
    """Configure the environment for a GPU experiment. Must be called before importing JAX."""
    if "jax" in sys.modules:
        raise RuntimeError("iadrift.runtime.setup() must run before JAX is imported.")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if theta_gradients:
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        # Otherwise the weight-gradient programs choose GPU kernels per process and identical runs can differ
        # (IADrift step-100 LR loss 49.49 vs 48.18 in one case).
        flag = "--xla_gpu_deterministic_ops=true"
        if flag not in os.environ.get("XLA_FLAGS", ""):
            os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + flag).strip()
    elif os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") == "platform":
        raise RuntimeError("Unset XLA_PYTHON_CLIENT_ALLOCATOR: CSGM does not reproduce the saved runs with 'platform'.")
    os.environ["HF_ROOT"] = str(paths.HF_CACHE_DIR)
    if not (paths.DRIFTING_DIR / "models" / "generator.py").exists():
        raise RuntimeError(f"Upstream drifting code not found in {paths.DRIFTING_DIR}; set DRIFTING_DIR (see README).")
    if str(paths.DRIFTING_DIR) not in sys.path:
        sys.path.insert(0, str(paths.DRIFTING_DIR))
    os.environ["IADRIFT_REQUIRE_GPU"] = "1" if require_gpu else "0"


def setup_cpu():
    """Force JAX onto the CPU, for analysis scripts (metrics, back-projection, statistics)."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
