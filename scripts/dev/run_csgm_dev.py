"""CSGM on the development images (development, 4 images; the starting point of every notebook 02 experiment).

Frozen CSGM (iadrift.csgm.FROZEN_CONFIG: Adam lr 0.02 on eps, 5,000 steps, no regularizer, 2 restarts from
key(2026, image, restart)) on SR x4:

    eps_hat = argmin_eps ||y - A G(eps; z, c)||^2,     x_hat = G(eps_hat; z, c)

Images: 4 development images (8117 streetcar, 3939 bonnet, 403 crane, 347 spoonbill), drawn with seed 2030 from the
100 images of the CSGM settings sweep and checked to be outside the test split. Labels per image: the correct class
("oracle") and one random wrong class (data.wrong_label_for, as in the test benchmark); for image 8117 also 5 more
random wrong classes (seed (2031, 8117); notebook 02's label experiment, files "label<c>"). Every label starts from
the same (eps, z). The restart with the lowest final LR loss is used downstream (never HR).

Ported from scripts/run_sr4_iadrift_dev.py (pick_dev_images, csgm_cached; it called run_sr4_csgm_test.reconstruct =
iadrift.csgm.CSGM.reconstruct) and notebooks/02_iadrift_dev_tuning.ipynb cell 15 (the 5 extra labels). The originals
ran in a process with XLA_PYTHON_CLIENT_ALLOCATOR=platform; notebook 04 Check 4 (default allocator) nevertheless
reproduced img08117_oracle_r1 bit for bit, so this runner uses the default allocator like every CSGM runner.

Usage:
    python scripts/dev/run_csgm_dev.py --dry-run     # plan and cache status; runs and writes nothing (CPU)
    python scripts/dev/run_csgm_dev.py --smoke       # rerun the first 500 steps of img08117_oracle_r1; writes nothing
    python scripts/dev/run_csgm_dev.py               # run what is missing (existing files are kept)

Outputs (IADRIFT_DATA/csgm_dev_5000steps_lr0.02/): img#####_{oracle,wrong,label<c>}_r{0,1}.npz with metrics_json
(initial/final LR loss, PSNR, eps statistics, class_label), eps, noise_labels, x_hat, trace_steps, trace_loss.
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
import json

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import load_npz, load_results, save_npz

CSGM_DIR = paths.DATA_DIR / "csgm_dev_5000steps_lr0.02"
DEV_SEED = 2030                 # the 4 development images, from the sweep's 100 images
SWEEP_RESULTS = "phase1_sr4_oracle_moments_dev_v1_6b22675bd557.jsonl"
LABEL_IMAGE, NUM_EXTRA_WRONG, LABEL_SEED = 8117, 5, 2031
NUM_RESTARTS = 2
SMOKE = {"image_index": 8117, "condition": "oracle", "restart": 1, "steps": 500}


def pick_dev_images(num_images=4):
    pool = sorted({row["image_index"] for row in load_results(paths.DATA_DIR / SWEEP_RESULTS)})
    return [int(i) for i in np.random.default_rng(DEV_SEED).choice(pool, size=num_images, replace=False)]


def check_dev(images, train_indices, test_indices):
    bad = [i for i in images if i not in set(train_indices.tolist()) or i in set(test_indices.tolist())]
    if bad:
        raise SystemExit(f"Not development images: {bad}.")


def build_plan(images, labels):
    """[(image, condition, restart, class label)]: oracle and wrong for every image, 5 extra labels for 8117."""
    plan = []
    for i in images:
        true_label = int(labels[i])
        conditions = [("oracle", true_label), ("wrong", data.wrong_label_for(i, true_label))]
        if i == LABEL_IMAGE:
            rng = np.random.default_rng([LABEL_SEED, LABEL_IMAGE])
            extra = rng.choice(np.setdiff1d(np.arange(1000), [true_label]), NUM_EXTRA_WRONG, replace=False)
            conditions += [(f"label{int(c)}", int(c)) for c in extra]
        plan += [(i, cond, r, label) for cond, label in conditions for r in range(NUM_RESTARTS)]
    return plan


def path_of(i, condition, restart):
    return CSGM_DIR / f"img{i:05d}_{condition}_r{restart}.npz"


def is_cached(i, condition, restart, label):
    """As the original csgm_cached: a file counts only if it was run with the same class label."""
    path = path_of(i, condition, restart)
    return path.exists() and load_npz(path)[0]["class_label"] == label


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; run and write nothing")
    parser.add_argument("--smoke", action="store_true", help="rerun the start of one saved run; writes nothing")
    parser.add_argument("--smoke-steps", type=int, default=SMOKE["steps"], help="5000 = the whole run")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train_indices, test_indices = data.load_split(labels)
    images = pick_dev_images()
    assert tuple(images) == data.DEV_IMAGES_4, images
    check_dev(images, train_indices, test_indices)
    plan = build_plan(images, labels)
    status = [is_cached(*run) for run in plan]

    if args.dry_run:
        print("DRY RUN: nothing is run or written.")
        print(f"Development images (seed {DEV_SEED}): {images}; restarts {NUM_RESTARTS}; Adam lr 0.02, 5,000 steps")
        for i, cond, r, label in plan:
            state = "cached" if is_cached(i, cond, r, label) else "MISSING"
            print(f"  img {i:5d}  {cond:<9} r{r}  class {label:3d}  {state}")
        print(f"Outputs present: {sum(status)} / {len(plan)} in {CSGM_DIR}")
        return

    from iadrift.csgm import CSGM
    from iadrift.generator import load_generator

    csgm = CSGM(load_generator())
    lr_all = data.load_measurements()

    if args.smoke:
        i, cond, r, k = SMOKE["image_index"], SMOKE["condition"], SMOKE["restart"], args.smoke_steps
        saved_m, saved = load_npz(path_of(i, cond, r))
        m, a = csgm.reconstruct(i, r, int(labels[i]), lr_all[i], data.hr_image(index_rows, i), num_steps=k)
        n = len(a["trace_loss"])
        ok = np.array_equal(a["trace_loss"], saved["trace_loss"][:n]) and np.array_equal(a["noise_labels"],
                                                                                         saved["noise_labels"])
        if k == csgm.num_steps:
            ok &= (m["final_loss"] == saved_m["final_loss"] and np.array_equal(a["eps"], saved["eps"])
                   and np.array_equal(a["x_hat"], saved["x_hat"]))
        print(f"[smoke] img {i} {cond} r{r}, {k} steps: loss trace (first {n} values) identical to the saved run: "
              f"{np.array_equal(a['trace_loss'], saved['trace_loss'][:n])}; last {a['trace_loss'][-1]!r} vs "
              f"{saved['trace_loss'][n - 1]!r}")
        print("SMOKE PASSED" if ok else "SMOKE FAILED")
        raise SystemExit(0 if ok else 1)

    for (i, cond, r, label), done in zip(plan, status):
        if done:
            continue
        hr = data.hr_image(index_rows, i)
        metrics, arrays = csgm.reconstruct(i, r, label, lr_all[i], hr)
        metrics["class_label"] = label
        save_npz(path_of(i, cond, r), arrays, metrics)
        print(f"img {i} {cond} r{r} (class {label}): loss {metrics['final_loss']:.2f}, PSNR {metrics['psnr']:.2f} dB, "
              f"{metrics['elapsed_seconds']:.0f} s", flush=True)
    print(f"ALL RUNS COMPLETE: {CSGM_DIR}")


if __name__ == "__main__":
    main()
