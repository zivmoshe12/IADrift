"""Generator samples and in-range targets for the 4 development classes.

Forward passes only:
1. Random samples x = G(eps; z, c, cfg) for the classes of images 8117, 3939, 403, 347. Two draws per class, (eps, z)
   from key fold_in(fold_in(2031, class), draw), at guidance 1.0. They show that the 16-pixel patch grid comes from the generator itself, not from the inversion.
2. The in-range SR targets x* = G(eps*; z*, c) of run_representation_dev.py (R3; key fold_in(2032, image), guidance
   1.0). That runner does not save x*.
Before saving, the saved R3 CSGM outputs are rescored against the regenerated x* and must give the PSNR stored in
representation_error_dev_v1/summary.json, so run_representation_dev.py has to run first.

Usage:
    python scripts/dev/run_generator_samples_dev.py --dry-run   # cache status; runs and writes nothing (CPU)
    python scripts/dev/run_generator_samples_dev.py --smoke     # regenerate 2 arrays, compare exactly; writes nothing
    python scripts/dev/run_generator_samples_dev.py             # create the file if it is missing

Output (IADRIFT_DATA/generator_samples_dev_v1.npz): sample_img<i>_draw<d>_cfg1 and xstar_img<i>, float32
[256, 256, 3] (12 arrays). Read by scripts/analysis/export_dev_results.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

import os

if "--dry-run" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from iadrift.runtime import setup

setup(theta_gradients=False)

import argparse
import io
import json

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import atomic_write_bytes, load_npz

OUT_PATH = paths.DATA_DIR / "generator_samples_dev_v1.npz"
REP_DIR = paths.DATA_DIR / "representation_error_dev_v1"
DEV_IMAGES = list(data.DEV_IMAGES_4)
SAMPLE_SEED, CFG_SCALES, SAMPLE_DRAWS = 2031, (1.0,), 2
INRANGE_SEED = 2032
SMOKE_KEYS = ("xstar_img403", "sample_img403_draw0_cfg1")


def expected_keys():
    keys = []
    for i in DEV_IMAGES:
        keys += [f"sample_img{i}_draw{d}_cfg{cfg:g}" for d in range(SAMPLE_DRAWS) for cfg in CFG_SCALES]
        keys.append(f"xstar_img{i}")
    return keys


def generate_all(gen, labels, only=None):
    """{name: float32 [256, 256, 3]}; only: optional set of names to compute (the others are skipped)."""
    import jax
    import jax.numpy as jnp

    def generate_cfg(eps, label, noise_labels, cfg):
        return np.asarray(gen.generate_cfg(eps, label, noise_labels, cfg).astype(jnp.float32))[0]

    out = {}
    for i in DEV_IMAGES:
        label = int(labels[i])
        for d in range(SAMPLE_DRAWS):
            key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(SAMPLE_SEED), label), d)
            k_eps, k_nl = jax.random.split(key)
            eps = jax.random.normal(k_eps, (1, 256, 256, 3), dtype=jnp.float32)
            nl = jax.random.randint(k_nl, (1, gen.noise_coords), 0, gen.noise_classes, dtype=jnp.int32)
            for cfg in CFG_SCALES:
                name = f"sample_img{i}_draw{d}_cfg{cfg:g}"
                if only is None or name in only:
                    out[name] = generate_cfg(eps, label, nl, cfg)
        name = f"xstar_img{i}"
        if only is None or name in only:
            k_eps, k_nl = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(INRANGE_SEED), i))
            eps_star = jax.random.normal(k_eps, (1, 256, 256, 3), dtype=jnp.float32)
            nl_star = jax.random.randint(k_nl, (1, gen.noise_coords), 0, gen.noise_classes, dtype=jnp.int32)
            out[name] = np.asarray(jax.device_get(gen.generate(eps_star, label, nl_star)), dtype=np.float32)[0]
    return out


def check_against_r3(out):
    """max |PSNR(saved kept R3 CSGM output, regenerated x*) - PSNR in summary.json| over images and conditions."""
    from iadrift.metrics import psnr

    summary = json.loads((REP_DIR / "summary.json").read_text())["R3"]
    worst = 0.0
    for i in DEV_IMAGES:
        for cond in ("noise labels known", "noise labels random"):
            runs = []
            for r in (0, 1):
                metrics, arrays = load_npz(REP_DIR / f"R3_img{i:05d}_{cond.replace(' ', '_')}_r{r}.npz")
                runs.append((metrics["final_loss"], r, arrays["x_hat"]))
            _, _, x = min(runs, key=lambda t: (t[0], t[1]))
            stored = summary[str(i)]["metrics"][f"CSGM, {cond}"]["psnr"]
            worst = max(worst, abs(psnr(out[f"xstar_img{i}"], x) - stored))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="cache status; run and write nothing")
    parser.add_argument("--smoke", action="store_true", help="regenerate two saved arrays; writes nothing")
    args = parser.parse_args()

    labels = data.labels_of(data.load_index())
    keys = expected_keys()

    if args.dry_run:
        print("DRY RUN: nothing is run or written.")
        present = []
        if OUT_PATH.exists():
            with np.load(OUT_PATH) as saved:
                present = [k for k in keys if k in saved.files]
        print(f"{len(DEV_IMAGES)} classes x {SAMPLE_DRAWS} draws x CFG {CFG_SCALES} "
              f"+ {len(DEV_IMAGES)} in-range targets")
        print(f"Outputs present: {len(present)} / {len(keys)} arrays in {OUT_PATH}")
        return

    from iadrift.generator import load_generator

    gen = load_generator()
    if args.smoke:
        out = generate_all(gen, labels, only=set(SMOKE_KEYS))
        with np.load(OUT_PATH) as saved:
            same = {k: bool(np.array_equal(out[k], saved[k])) for k in SMOKE_KEYS}
        print(f"[smoke] identical to the saved arrays: {same}")
        print("SMOKE PASSED" if all(same.values()) else "SMOKE FAILED")
        raise SystemExit(0 if all(same.values()) else 1)

    if OUT_PATH.exists():
        print(f"{OUT_PATH} exists; nothing to do.")
        return
    out = generate_all(gen, labels)
    worst = check_against_r3(out)
    print(f"Check: max |PSNR(saved R3 CSGM output, regenerated x*) - stored| = {worst:.2e} dB")
    assert worst < 1e-6, "regenerated targets differ from the saved ones"
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **out)
    atomic_write_bytes(OUT_PATH, buffer.getvalue())
    print(f"Saved {len(out)} arrays to {OUT_PATH}")


if __name__ == "__main__":
    main()
