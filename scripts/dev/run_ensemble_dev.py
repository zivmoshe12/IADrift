"""Shifted CSGM members of the development posterior-mean ensemble (6 development images).

Members per image, correct label (iadrift.ensemble): the 2 CSGM restarts r0, r1, made by other runners, and 3 CSGM
runs on the measurement circularly shifted by s in {(0, 1), (1, 0), (1, 1)} LR pixels. The shifted runs all start
from initialize(i, 0) and are later shifted back by 4s HR pixels:

    x_s = S_{-4s} CSGM(S_s y),   S_s y = jnp.roll(y, s, axis=(1, 2))

CSGM is frozen (Adam lr 0.02, 5,000 steps, loss ||S_s y - A G(eps)||^2), run through CSGM.run with the identity
projection, which reproduces the frozen loop bit for bit (see run_eps_projection_dev.py). This script only makes the
shifted members. Averaging, weighting (w_k ~ exp(-(L_k - L_min) / std L), L_k the LR loss), back-projection and
scoring are in scripts/analysis/export_ablation_results.py.

Images: 8117, 3939, 403, 347, 9780, 8405. The shifted runs of image 403 were made first, as a one-image demo, and
live in one_image_demos_v1/; the other images' runs are in ensemble_dev_v1/.
Restart members: csgm_dev_5000steps_lr0.02/img#####_oracle_r{0,1}.npz for the first 4 images (run_csgm_dev.py) and
z_swap_dev_v1/confirm/img#####_csgm_r{0,1}.npz for 9780 and 8405 (run_z_swap_confirm_dev.py).

Usage:
    python scripts/dev/run_ensemble_dev.py --dry-run   # plan and cache status (incl. restart members); writes nothing
    python scripts/dev/run_ensemble_dev.py --smoke     # GPU: 100 steps of one shifted run; trace prefix
    python scripts/dev/run_ensemble_dev.py             # run the missing shifted members

Outputs (IADRIFT_DATA): ensemble_dev_v1/img#####_shift##_r0.npz and one_image_demos_v1/D_shift##_r0.npz (image 403);
metrics_json {final_loss, seconds}; eps, noise_labels, x_hat (in the shifted frame), trace_steps, trace_loss.
Read by scripts/analysis/export_ablation_results.py (section 5, ensembles).
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

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import load_npz, save_npz
from iadrift.ensemble import SHIFTS, START_RESTART

ENS_DIR = paths.DATA_DIR / "ensemble_dev_v1"
DEMO_DIR = paths.DATA_DIR / "one_image_demos_v1"
CSGM_CACHE = paths.DATA_DIR / "csgm_dev_5000steps_lr0.02"
CONFIRM_DIR = paths.DATA_DIR / "z_swap_dev_v1" / "confirm"
IMAGES = list(data.DEV_IMAGES_6)
DEMO_IMAGE = 403
SMOKE = (9780, (0, 1), 100)


def shift_path(i, sy, sx):
    return DEMO_DIR / f"D_shift{sy}{sx}_r0.npz" if i == DEMO_IMAGE else ENS_DIR / f"img{i:05d}_shift{sy}{sx}_r0.npz"


def restart_path(i, r):
    if i in data.DEV_IMAGES_4:
        return CSGM_CACHE / f"img{i:05d}_oracle_r{r}.npz"
    return CONFIRM_DIR / f"img{i:05d}_csgm_r{r}.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; runs and writes nothing")
    parser.add_argument("--smoke", action="store_true", help="GPU: 100 steps of one shifted run; writes nothing")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train, test = data.load_split(labels)
    assert set(IMAGES) <= set(train.tolist()) and not set(IMAGES) & set(test.tolist()), "development images only"
    lr_all = data.load_measurements()
    plan = [(i, sy, sx) for i in IMAGES for sy, sx in SHIFTS]

    if args.dry_run:
        restarts = [restart_path(i, r) for i in IMAGES for r in (0, 1)]
        print("DRY RUN: nothing is run or written.")
        print(f"Images {IMAGES}; shifts {SHIFTS} (LR pixels) from initialize(i, {START_RESTART}); "
              "CSGM lr 0.02, 5,000 steps")
        print(f"Shifted members: {sum(shift_path(*p).exists() for p in plan)} / {len(plan)} present "
              f"({ENS_DIR.name}/, {DEMO_DIR.name}/ for image {DEMO_IMAGE})")
        print(f"Restart members (other runners): {sum(p.exists() for p in restarts)} / {len(restarts)} present")
        return

    import jax.numpy as jnp
    from iadrift.csgm import CSGM, project_none
    from iadrift.generator import load_generator

    csgm = CSGM(load_generator())

    def shifted_run(i, sy, sx, num_steps=None):
        y = jnp.asarray(lr_all[i:i + 1], jnp.float32)
        eps0, z0 = csgm.initialize(i, START_RESTART)
        return csgm.run(csgm.measurement_loss, project_none, jnp.roll(y, (sy, sx), axis=(1, 2)), int(labels[i]),
                        eps0, z0, num_steps=num_steps)

    if args.smoke:
        i, (sy, sx), steps = SMOKE
        _, a = shifted_run(i, sy, sx, steps)
        _, saved = load_npz(shift_path(i, sy, sx))
        n = len(a["trace_loss"])
        same = bool(np.array_equal(a["trace_loss"], saved["trace_loss"][:n]))
        print(f"{shift_path(i, sy, sx).name}: first {steps} steps, trace {a['trace_loss'].tolist()} vs saved "
              f"{saved['trace_loss'][:n].tolist()}; identical: {same}")
        print("SMOKE PASSED" if same else "SMOKE FAILED")
        raise SystemExit(0 if same else 1)

    for i, sy, sx in plan:
        path = shift_path(i, sy, sx)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        m, a = shifted_run(i, sy, sx)
        save_npz(path, a, m)
        print(f"image {i} shift ({sy}, {sx}): final LR loss {m['final_loss']:.1f} ({m['seconds']:.0f} s)", flush=True)
    missing = [p for p in (restart_path(i, r) for i in IMAGES for r in (0, 1)) if not p.exists()]
    if missing:
        print("Restart members still missing (run_csgm_dev.py / run_z_swap_confirm_dev.py):", missing)
    else:
        print("ALL RUNS COMPLETE")


if __name__ == "__main__":
    main()
