"""CSGM on the two extra development images of the ensemble experiments (9780, 8405).

Frozen CSGM (CSGM.reconstruct): initialize(i, r) for restarts 0 and 1, Adam lr 0.02, 5,000 steps, correct label.
These runs are the restart members r0, r1 of the development ensembles for 9780 and 8405; the 4 other development
images get theirs from run_csgm_dev.py. The images are the first 2 draws of pick_dev_images(6) (numpy seed 2030 over
the 100-image pool of run_csgm_settings_sweep.py) that are not among those 4.

Usage:
    python scripts/dev/run_csgm_extra_dev.py --dry-run   # plan and cache status; runs and writes nothing (CPU)
    python scripts/dev/run_csgm_extra_dev.py --smoke     # GPU: rerun 100 steps of 9780 r0, compare the trace
    python scripts/dev/run_csgm_extra_dev.py             # run what is missing (resumable)

Outputs (IADRIFT_DATA/z_swap_dev_v1/confirm/, a cache name kept from the original runs): img#####_csgm_r{0,1}.npz
(CSGM.reconstruct metrics + class_label; eps, noise_labels, x_hat, trace_steps, trace_loss). Read by the ensemble
runners and scripts/analysis/export_ablation_results.py.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

if "--dry-run" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from iadrift.runtime import setup

setup(theta_gradients=False, require_gpu="--dry-run" not in sys.argv)  # default allocator (CSGM numerics)

import argparse
import json

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import load_npz, save_npz
from iadrift.csgm import FROZEN_CONFIG

OUT_DIR = paths.DATA_DIR / "z_swap_dev_v1" / "confirm"
DEV_SEED = 2030
DEV_POOL_PATH = paths.DATA_DIR / "phase1_sr4_oracle_moments_dev_v1_6b22675bd557.jsonl"  # the settings-sweep pool
USED = list(data.DEV_IMAGES_4)
NUM_NEW = 2
RESTARTS = range(FROZEN_CONFIG["num_restarts"])
SMOKE_IMAGE = 9780


def pick_dev_images(num_images):
    """A seeded draw from the settings sweep's image pool (same seed and pool as run_csgm_dev.pick_dev_images)."""
    pool = sorted({json.loads(line)["image_index"] for line in DEV_POOL_PATH.open() if line.strip()})
    return [int(i) for i in np.random.default_rng(DEV_SEED).choice(pool, size=num_images, replace=False)]


def new_images(train, test):
    new = [i for i in pick_dev_images(len(USED) + NUM_NEW) if i not in USED][:NUM_NEW]
    assert set(new) <= set(train.tolist()) and not set(new) & set(test.tolist()), "development images only"
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; runs and writes nothing")
    parser.add_argument("--smoke", action="store_true", help="GPU: rerun 100 steps of one saved run; no writes")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train, test = data.load_split(labels)
    new = new_images(train, test)
    lr_all = data.load_measurements()
    print("development images:", new, flush=True)
    files = [OUT_DIR / f"img{i:05d}_csgm_r{r}.npz" for i in new for r in RESTARTS]

    if args.dry_run:
        print("DRY RUN: nothing is run or written.")
        print(f"CSGM, {FROZEN_CONFIG['num_steps']:,} steps, restarts {list(RESTARTS)}")
        print(f"Cache {OUT_DIR}: {sum(p.exists() for p in files)} / {len(files)} outputs present")
        return

    from iadrift.csgm import CSGM
    from iadrift.generator import load_generator

    csgm = CSGM(load_generator())
    if args.smoke:
        i, r = SMOKE_IMAGE, 0
        _, a = csgm.reconstruct(i, r, int(labels[i]), lr_all[i], data.hr_image(index_rows, i), num_steps=100)
        _, saved = load_npz(OUT_DIR / f"img{i:05d}_csgm_r{r}.npz")
        ok = bool(np.array_equal(a["trace_loss"], saved["trace_loss"][:len(a["trace_loss"])]))
        print(f"CSGM image {i} r{r}, 100 steps: trace prefix identical: {ok}")
        print("SMOKE PASSED" if ok else "SMOKE FAILED")
        raise SystemExit(0 if ok else 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in new:
        label = int(labels[i])
        hr = data.hr_image(index_rows, i)
        for r in RESTARTS:
            path = OUT_DIR / f"img{i:05d}_csgm_r{r}.npz"
            if not path.exists():
                m, a = csgm.reconstruct(i, r, label, lr_all[i], hr)
                m["class_label"] = label
                save_npz(path, a, m)
            m = load_npz(path)[0]
            print(f"CSGM image {i} r{r}: LR loss {m['final_loss']:.1f}, PSNR {m['psnr']:.2f} dB", flush=True)
    print("ALL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
