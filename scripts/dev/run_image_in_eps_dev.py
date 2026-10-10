"""Feeding an image instead of noise as eps (3 development images, forward passes only).

For images 403, 3939, 8405 (correct label, the z of CSGM restart 0 = initialize(i, 0)[1], CFG 1.0), compute the
generator output x = G(eps_in; z, c) for

    eps_in = alpha s + sqrt(1 - alpha^2) n,   alpha in {0, 0.25, 0.5, 0.75, 1},

where s is the per-channel standardized input image (bicubic(y), or HR as an oracle diagnostic) and n one fixed noise
draw normal(PRNGKey(2040 + i), (256, 256, 3)); plus the raw, unstandardized image as eps. Nothing is optimized or
chosen. Before saving, the PSNRs are checked against the values recorded when the experiment was first run (SAVED_PSNR,
1 decimal).

Usage:
    python scripts/dev/run_image_in_eps_dev.py --dry-run   # plan and cache status; runs and writes nothing (CPU)
    python scripts/dev/run_image_in_eps_dev.py --smoke     # GPU: regenerate 403|bicubic(y)|a0.5, compare bit-exactly
    python scripts/dev/run_image_in_eps_dev.py             # regenerate all 36 outputs if the file is missing

Outputs (IADRIFT_DATA/image_in_eps_dev_v1.npz): 36 float32 [256, 256, 3] arrays "<image>|<input>|<a0..a1 or raw>".
Used in the full report, Development ablations.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

if "--dry-run" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from iadrift.runtime import setup

setup(theta_gradients=False, require_gpu="--dry-run" not in sys.argv)  # forward passes; default allocator

import argparse
import io

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import atomic_write_bytes

OUT_PATH = paths.DATA_DIR / "image_in_eps_dev_v1.npz"
IMAGES, ALPHAS = [403, 3939, 8405], [0.0, 0.25, 0.5, 0.75, 1.0]
SOURCES = ("bicubic(y)", "HR (oracle)")
NOISE_SEED = 2040  # + image index
# PSNR recorded when the experiment was first run (dB, 1 decimal): {image: {input: [alpha 0, .25, .5, .75, 1, raw]}}
SAVED_PSNR = {403: {"bicubic(y)": [8.5, 8.5, 7.9, 9.2, 11.2, 10.3], "HR (oracle)": [8.5, 8.0, 9.0, 8.7, 7.2, 10.2]},
             3939: {"bicubic(y)": [7.2, 7.1, 7.2, 7.4, 7.3, 7.5], "HR (oracle)": [7.2, 7.3, 7.0, 7.6, 7.5, 7.5]},
             8405: {"bicubic(y)": [6.9, 6.4, 5.7, 5.5, 4.1, 4.1], "HR (oracle)": [6.9, 6.4, 5.6, 5.5, 4.5, 4.0]}}
SMOKE_KEY = "403|bicubic(y)|a0.5"


def keys():
    return [f"{i}|{src}|{a}" for i in IMAGES for src in SOURCES for a in [f"a{x:g}" for x in ALPHAS] + ["raw"]]


def std_img(x):
    x = np.asarray(x, np.float32)
    return (x - x.mean(axis=(0, 1))) / x.std(axis=(0, 1))


def generate_all(gen, index_rows, lr_all, only=None):
    """{key: output} and the worst PSNR deviation from SAVED_PSNR; `only` restricts to one key."""
    import jax
    import jax.numpy as jnp
    from iadrift.metrics import psnr
    from iadrift.operators import bicubic_up

    out, worst = {}, 0.0
    for i in IMAGES:
        if only and not only.startswith(f"{i}|"):
            continue
        label = int(index_rows[i]["label"])
        hr = data.hr_image(index_rows, i)
        bic = bicubic_up(lr_all[i])
        _, z0 = gen.initialize(i, 0)
        n = np.asarray(jax.random.normal(jax.random.PRNGKey(NOISE_SEED + i), (256, 256, 3)), np.float32)
        for src, img in (("bicubic(y)", bic), ("HR (oracle)", hr)):
            s = std_img(img)
            inputs = [a * s + np.sqrt(1 - a ** 2) * n for a in ALPHAS] + [np.asarray(img, np.float32)]
            for k, e in enumerate(inputs):
                tag = f"a{ALPHAS[k]:g}" if k < len(ALPHAS) else "raw"
                if only and only != f"{i}|{src}|{tag}":
                    continue
                x = np.asarray(gen.generate(jnp.asarray(e)[None], label, z0).astype(jnp.float32))[0]
                out[f"{i}|{src}|{tag}"] = x
                worst = max(worst, abs(round(psnr(hr, x), 1) - SAVED_PSNR[i][src][k]))
    return out, worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; runs and writes nothing")
    parser.add_argument("--smoke", action="store_true", help="GPU: regenerate one output and compare; writes nothing")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train, test = data.load_split(labels)
    assert set(IMAGES) <= set(train.tolist()) and not set(IMAGES) & set(test.tolist()), "development images only"
    lr_all = data.load_measurements()

    if args.dry_run:
        print("DRY RUN: nothing is run or written.")
        print(f"Images {IMAGES}; inputs {SOURCES}; alphas {ALPHAS} + raw; noise PRNGKey({NOISE_SEED} + i)")
        found = 0
        if OUT_PATH.exists():
            with np.load(OUT_PATH) as d:
                found = len(set(d.files) & set(keys()))
        print(f"Cache {OUT_PATH}: {found} / {len(keys())} outputs present")
        return

    from iadrift.generator import load_generator

    gen = load_generator()
    if args.smoke:
        out, _ = generate_all(gen, index_rows, lr_all, only=SMOKE_KEY)
        with np.load(OUT_PATH) as d:
            same = bool(np.array_equal(out[SMOKE_KEY], d[SMOKE_KEY]))
        print(f"{SMOKE_KEY}: identical to the cache: {same}")
        print("SMOKE PASSED" if same else "SMOKE FAILED")
        raise SystemExit(0 if same else 1)

    if OUT_PATH.exists():
        print(f"{OUT_PATH} exists; nothing to do.\nALL RUNS COMPLETE")
        return
    out, worst = generate_all(gen, index_rows, lr_all)
    print(f"Check: max |PSNR (1 decimal) - recorded| = {worst:.2f} dB")
    assert worst < 0.11, "regenerated outputs differ from the saved ones"
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **out)
    atomic_write_bytes(OUT_PATH, buffer.getvalue())
    print(f"Saved {len(out)} outputs to {OUT_PATH}\nALL RUNS COMPLETE")


if __name__ == "__main__":
    main()
