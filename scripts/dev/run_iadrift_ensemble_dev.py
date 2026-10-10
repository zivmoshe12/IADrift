"""IADrift from every member of the development CSGM ensemble (development, 6 images; notebook 04 §5).

Check 14. For each of the 6 development images (8117, 3939, 403, 347, 9780, 8405; correct label) and each CSGM ensemble
member (restarts r0, r1 and the r0-start runs on the measurement shifted by (0, 1), (1, 0), (1, 1) LR pixels; see
run_ensemble_dev.py), IADrift exactly as frozen (docs/PROTOCOL.md §4; iadrift.adaptation): joint Adam on all generator
weights theta (lr 1e-5) and on eps (lr 1e-3) from that member's (eps, z), 1,000 steps, against that member's
measurement (shifted for shift members),
    min_{theta, eps} ||S_s y - A G_theta(eps; z, c)||^2,
saving the eager outputs at steps 100 and 1,000 in the member's own (possibly shifted) frame. Averaging, un-shifting,
back-projection and scoring are done by scripts/analysis/export_ablation_results.py (weights from the CSGM members'
LR losses, never HR). The printed PSNR uses the HR image shifted into the member's frame (logging only).

Ported from scripts/run_iadrift_ensemble_dev.py.

Usage:
    python scripts/dev/run_iadrift_ensemble_dev.py --dry-run   # plan and cache status; runs and writes nothing (CPU)
    python scripts/dev/run_iadrift_ensemble_dev.py --smoke     # GPU: 100 steps for 403/r0, compare x_step100 exactly
    python scripts/dev/run_iadrift_ensemble_dev.py             # run what is missing (one file per image and member)

Outputs (IADRIFT_DATA/iadrift_ensemble_dev_v1/img#####_<member>.npz): metrics_json {image, member, shift, source,
step100/step1000: {lr_loss, psnr, finite}, seconds}; x_step100, x_step1000.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

if "--dry-run" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from iadrift.runtime import setup

setup(theta_gradients=True, require_gpu="--dry-run" not in sys.argv)  # gradients w.r.t. theta: platform allocator

import argparse
from time import perf_counter

import numpy as np

from iadrift import data, paths
from iadrift.adaptation import FROZEN_CONFIG
from iadrift.checkpoint import load_npz, save_npz
from iadrift.ensemble import MEMBERS, SHIFTS

OUT_DIR = paths.DATA_DIR / "iadrift_ensemble_dev_v1"
CSGM_CACHE = paths.DATA_DIR / "csgm_dev_5000steps_lr0.02"
CONFIRM = paths.DATA_DIR / "z_swap_dev_v1" / "confirm"
ENS = paths.DATA_DIR / "ensemble_dev_v1"
DEMO = paths.DATA_DIR / "one_image_demos_v1"
IMAGES = list(data.DEV_IMAGES_6)
CKPTS = tuple(FROZEN_CONFIG["checkpoints"])  # (100, 1000)
SMOKE = (403, "r0")


def members(i):
    """{member: (source CSGM file, shift in LR pixels)}, in the order of iadrift.ensemble.MEMBERS."""
    m = {}
    for r in (0, 1):
        m[f"r{r}"] = (CSGM_CACHE / f"img{i:05d}_oracle_r{r}.npz" if i in data.DEV_IMAGES_4
                      else CONFIRM / f"img{i:05d}_csgm_r{r}.npz", (0, 0))
    for sy, sx in SHIFTS:
        src = DEMO / f"D_shift{sy}{sx}_r0.npz" if i == 403 else ENS / f"img{i:05d}_shift{sy}{sx}_r0.npz"
        m[f"shift{sy}{sx}"] = (src, (sy, sx))
    assert tuple(m) == MEMBERS
    return m


def run_member(adapt, steps, i, label, hr, y0, src, shift, num_steps=CKPTS[-1]):
    """IADrift from one member (the original loop). Returns (info, outputs)."""
    import jax.numpy as jnp

    opt_t, opt_e, step = steps
    sy, sx = shift
    with np.load(src) as d:
        eps, z = jnp.asarray(d["eps"]), jnp.asarray(d["noise_labels"], jnp.int32).reshape(1, -1)
    y = jnp.roll(y0, (sy, sx), axis=(1, 2))
    hr_s = np.roll(hr, (4 * sy, 4 * sx), axis=(0, 1))  # only for the printed PSNR (in member coordinates)
    theta = adapt.fresh_weights()
    st_t, st_e = opt_t.init(theta), opt_e.init(eps)
    out, info = {}, {}
    t0 = perf_counter()
    for s in range(1, num_steps + 1):
        theta, eps, st_t, st_e, _ = step(theta, eps, st_t, st_e, y, label, z)
        if s in CKPTS:
            x, l, p = adapt.eager_metrics(theta, eps, label, z, y, hr_s)
            out[f"x_step{s}"] = np.asarray(x, np.float32)
            info[f"step{s}"] = {"lr_loss": l, "psnr": p, "finite": bool(np.all(np.isfinite(x)))}
    info["seconds"] = perf_counter() - t0
    del theta, st_t, st_e
    return info, out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; runs and writes nothing")
    parser.add_argument("--smoke", action="store_true", help="GPU: 100 steps of one member; writes nothing")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train, test = data.load_split(labels)
    assert set(IMAGES) <= set(train.tolist()) and not set(IMAGES) & set(test.tolist()), "development images only"
    lr_all = data.load_measurements()
    plan = [(i, name) for i in IMAGES for name in MEMBERS]

    if args.dry_run:
        sources = [src for i in IMAGES for src, _ in members(i).values()]
        print("DRY RUN: nothing is run or written.")
        print(f"Images {IMAGES}; members {MEMBERS}; IADrift theta lr {FROZEN_CONFIG['lr_theta']}, eps lr "
              f"{FROZEN_CONFIG['lr_eps']}, outputs at steps {CKPTS}")
        print(f"CSGM member files (inputs): {sum(p.exists() for p in sources)} / {len(sources)} present")
        print(f"Cache {OUT_DIR}: {sum((OUT_DIR / f'img{i:05d}_{n}.npz').exists() for i, n in plan)} / {len(plan)} "
              f"outputs present")
        return

    import jax.numpy as jnp
    from iadrift.adaptation import Adaptation
    from iadrift.generator import load_generator

    adapt = Adaptation(load_generator())
    steps = adapt.make_joint_step(FROZEN_CONFIG["lr_theta"], FROZEN_CONFIG["lr_eps"])

    def inputs(i):
        return int(labels[i]), data.hr_image(index_rows, i), jnp.asarray(lr_all[i:i + 1], jnp.float32)

    if args.smoke:
        i, name = SMOKE
        src, shift = members(i)[name]
        info, out = run_member(adapt, steps, i, *inputs(i), src, shift, num_steps=CKPTS[0])
        saved_m, saved = load_npz(OUT_DIR / f"img{i:05d}_{name}.npz")
        same = bool(np.array_equal(out["x_step100"], saved["x_step100"])) and \
            info["step100"]["lr_loss"] == saved_m["step100"]["lr_loss"]
        print(f"image {i} {name}, step 100: LR loss {info['step100']['lr_loss']!r} vs saved "
              f"{saved_m['step100']['lr_loss']!r}; output identical: {same} "
              f"(max |difference| {float(np.max(np.abs(out['x_step100'] - saved['x_step100']))):.2e})")
        print("SMOKE PASSED" if same else "SMOKE FAILED")
        raise SystemExit(0 if same else 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in IMAGES:
        label, hr, y0 = inputs(i)
        for name, (src, (sy, sx)) in members(i).items():
            path = OUT_DIR / f"img{i:05d}_{name}.npz"
            if path.exists():
                continue
            info, out = run_member(adapt, steps, i, label, hr, y0, src, (sy, sx))
            info = {"image": i, "member": name, "shift": [sy, sx], "source": src.name, **info}
            save_npz(path, out, info)
            print(f"image {i} {name}: LR loss {info['step100']['lr_loss']:.1f} (100) / "
                  f"{info['step1000']['lr_loss']:.1f} (1000); PSNR {info['step100']['psnr']:.2f} / "
                  f"{info['step1000']['psnr']:.2f} dB; {info['seconds']:.0f} s", flush=True)
    print("ALL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
