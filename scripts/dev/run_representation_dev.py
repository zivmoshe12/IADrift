"""Representation error of the frozen generator (development, 4 images; notebook 02, CSGM-HR and in-range SR rows).

Diagnostics, not methods. Both use the frozen CSGM loop (Adam lr 0.02 on eps, 5,000 steps, restarts r0 and r1 from
key(2026, image, restart), keep the restart with the lower final loss, no projection):

R1, CSGM fitted to HR (A = identity; uses HR on purpose, the best the frozen generator can do for the image):
    eps_hat = argmin_eps ||HR - G(eps; z, c)||^2
R3, in-range SR x4 (the target is in the generator's range, so the representation error is zero):
    x* = G(eps*; z*, c), y* = A x*, with eps* ~ N(0, I), z* random (key fold_in(2032, image)); then CSGM SR from y*
    with z = z* ("noise labels known") or z from initialize ("noise labels random", the real setting), scored
    against x*, plus ||eps - eps*|| / ||eps*||.
Images 8117, 3939, 403, 347, correct class. summary.json holds the metrics of the kept restarts (PSNR, LPIPS, LR loss,
band SNR; eps statistics for R1; for R3 also bicubic(y*) and bicubic + BP, the eps error and the loss at eps*).

Ported from notebooks/04_postproc_controls_dev.ipynb Check 5, cells 42 (R1) and 44 (R3), with the generic CSGM loop
of cell 37 (= iadrift.csgm.CSGM.run) and eps_stats of cell 32. Default allocator, as the original notebook (whose
Check 4 reproduced a saved CSGM run bit for bit with this loop).

Usage:
    python scripts/dev/run_representation_dev.py --dry-run     # plan and cache status; runs and writes nothing (CPU)
    python scripts/dev/run_representation_dev.py --smoke       # rerun 500 steps of R1 and R3 runs, compare the traces
    python scripts/dev/run_representation_dev.py               # run what is missing; summary.json if it is missing

Outputs (IADRIFT_DATA/representation_error_dev_v1/): R1_img#####_r{0,1}.npz and
R3_img#####_noise_labels_{known,random}_r{0,1}.npz (metrics_json: final_loss, seconds; eps, noise_labels, x_hat,
trace_steps, trace_loss), summary.json.
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
from iadrift.checkpoint import atomic_write_bytes, load_npz, save_npz

REP_DIR = paths.DATA_DIR / "representation_error_dev_v1"
DEV_IMAGES = list(data.DEV_IMAGES_4)
RESTARTS = (0, 1)
INRANGE_SEED = 2032
R3_CONDITIONS = ("noise labels known", "noise labels random")
SMOKE_STEPS = 500


def r1_path(i, r):
    return REP_DIR / f"R1_img{i:05d}_r{r}.npz"


def r3_path(i, cond, r):
    return REP_DIR / f"R3_img{i:05d}_{cond.replace(' ', '_')}_r{r}.npz"


def all_paths():
    return ([r1_path(i, r) for i in DEV_IMAGES for r in RESTARTS]
            + [r3_path(i, c, r) for i in DEV_IMAGES for c in R3_CONDITIONS for r in RESTARTS])


def eps_stats(e, e0=None):
    """Per-patch statistics of eps in the generator's patch order (16x16 grid of 16x16x3 patches)."""
    def patch_view(v):
        return np.asarray(v, np.float64).reshape(16, 16, 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(256, 768)

    p = patch_view(e)
    pn = np.linalg.norm(p, axis=1) / np.sqrt(768)
    s = {"rms": float(np.sqrt(np.mean(p ** 2))), "mean": float(p.mean()),
         "patch_norm_mean": float(pn.mean()), "patch_norm_std": float(pn.std()),
         "patch_norm_min": float(pn.min()), "patch_norm_max": float(pn.max()),
         "patch_mean_abs": float(np.abs(p.mean(1)).mean())}
    if e0 is not None:
        s["rel_change"] = float(np.linalg.norm(p - patch_view(e0)) / np.linalg.norm(patch_view(e0)))
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; run and write nothing")
    parser.add_argument("--smoke", action="store_true", help="rerun the start of two saved runs; writes nothing")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    _, test_indices = data.load_split(labels)
    assert not np.any(np.isin(DEV_IMAGES, test_indices)), "development images only"

    if args.dry_run:
        print("DRY RUN: nothing is run or written.")
        for p in all_paths():
            print(f"  {p.name:<45} {'cached' if p.exists() else 'MISSING'}")
        summary = REP_DIR / "summary.json"
        print(f"  {summary.name:<45} {'present' if summary.exists() else 'MISSING'}")
        present = sum(p.exists() for p in all_paths()) + summary.exists()
        print(f"Outputs present: {present} / {len(all_paths()) + 1} in {REP_DIR}")
        return

    import jax
    import jax.numpy as jnp

    from iadrift.csgm import CSGM, best_restart, project_none
    from iadrift.generator import load_generator
    from iadrift.operators import A_sr4

    gen = load_generator()
    csgm = CSGM(gen)
    generate, measurement_loss = gen.generate, csgm.measurement_loss
    lr_all = data.load_measurements()

    def hr_loss(eps, target, label, noise_labels):
        return jnp.sum((generate(eps, label, noise_labels).astype(jnp.float32) - target) ** 2)

    def in_range_target(i, label):
        k_eps, k_nl = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(INRANGE_SEED), i))
        eps_star = jax.random.normal(k_eps, (1, 256, 256, 3), dtype=jnp.float32)
        nl_star = jax.random.randint(k_nl, (1, gen.noise_coords), 0, gen.noise_classes, dtype=jnp.int32)
        x_star = np.asarray(jax.device_get(generate(eps_star, label, nl_star)), dtype=np.float32)[0]
        return eps_star, nl_star, x_star, A_sr4(jnp.asarray(x_star)[None])

    def cached_run(path, loss_fn, target, label, eps0, noise_labels, num_steps=None):
        if path.exists() and num_steps is None:
            return load_npz(path)
        metrics, arrays = csgm.run(loss_fn, project_none, target, label, eps0, noise_labels, num_steps)
        if num_steps is None:
            save_npz(path, arrays, metrics)
        return metrics, arrays

    if args.smoke:
        i, r = 8117, 0
        label = int(labels[i])
        hr_target = jnp.asarray(data.hr_image(index_rows, i), jnp.float32)[None]
        _, nl_star, _, y_star = in_range_target(i, label)
        eps0, nl0 = gen.initialize(i, r)
        ok = True
        for name, path, loss_fn, target, nl in (("R1", r1_path(i, r), hr_loss, hr_target, nl0),
                                               ("R3 known", r3_path(i, R3_CONDITIONS[0], r), measurement_loss,
                                                y_star, nl_star)):
            _, a = csgm.run(loss_fn, project_none, target, label, eps0, nl, SMOKE_STEPS)
            saved = load_npz(path)[1]["trace_loss"]
            same = np.array_equal(a["trace_loss"], saved[:len(a["trace_loss"])])
            ok &= same
            print(f"[smoke] {name}, image {i}, r{r}, {SMOKE_STEPS} steps: loss trace identical: {same} "
                  f"(last {a['trace_loss'][-1]!r} vs {saved[len(a['trace_loss']) - 1]!r})")
        print("SMOKE PASSED" if ok else "SMOKE FAILED")
        raise SystemExit(0 if ok else 1)

    r1, r3 = {}, {}
    for i in DEV_IMAGES:
        label, hr = int(labels[i]), data.hr_image(index_rows, i)
        target = jnp.asarray(hr, jnp.float32)[None]
        runs = {r: cached_run(r1_path(i, r), hr_loss, target, label, *gen.initialize(i, r)) for r in RESTARTS}
        r1[i] = best_restart(runs)
        print(f"R1 image {i}: final HR loss " + ", ".join(f"r{r} {runs[r][0]['final_loss']:.1f}" for r in RESTARTS)
              + f" -> keep r{r1[i][0]}", flush=True)

        eps_star, nl_star, x_star, y_star = in_range_target(i, label)
        r3[i] = {"eps_star": eps_star, "nl_star": nl_star, "x_star": x_star, "y_star": y_star}
        for cond in R3_CONDITIONS:
            runs = {}
            for r in RESTARTS:
                eps0, nl0 = gen.initialize(i, r)
                runs[r] = cached_run(r3_path(i, cond, r), measurement_loss, y_star, label, eps0,
                                     nl_star if cond == "noise labels known" else nl0)
            r3[i][cond] = best_restart(runs)
            print(f"R3 image {i}, {cond}: final LR loss " + ", ".join(
                f"r{r} {runs[r][0]['final_loss']:.2f}" for r in RESTARTS) + f" -> keep r{r3[i][cond][0]}", flush=True)

    summary_path = REP_DIR / "summary.json"
    if not summary_path.exists():
        write_summary(summary_path, r1, r3, index_rows, labels, lr_all, gen, measurement_loss)
    print(f"ALL RUNS COMPLETE: {REP_DIR}")


def write_summary(path, r1, r3, index_rows, labels, lr_all, gen, measurement_loss):
    """Notebook 04's summary: metrics of the kept restarts (as in cells 8, 42 and 44)."""
    import jax.numpy as jnp
    import torch

    from iadrift.metrics import band_snr_db, lpips_dist, psnr
    from iadrift.operators import A_sr4, backproject, bicubic_up

    torch.set_num_threads(4)  # as the notebook (LPIPS on the CPU)

    def evaluate(x, ref, y):  # y: [1, 64, 64, 3]
        return {"psnr": psnr(ref, x), "psnr_clipped": psnr(ref, np.clip(x, -1, 1)), "lpips": lpips_dist(x, ref),
                "lr_loss": float(jnp.sum((A_sr4(jnp.asarray(x, jnp.float32)[None]) - y) ** 2)),
                "band_snr_db": band_snr_db(x, ref)}

    summary = {"R1": {}, "R3": {}}
    for i in DEV_IMAGES:
        label, hr = int(labels[i]), data.hr_image(index_rows, i)
        y = jnp.asarray(lr_all[i:i + 1], jnp.float32)
        r, _, a = r1[i]
        summary["R1"][str(i)] = {"restart": r, "metrics": evaluate(a["x_hat"], hr, y),
                                 "eps_stats": eps_stats(a["eps"], np.asarray(gen.initialize(i, r)[0]))}
        t = r3[i]
        bic = bicubic_up(np.asarray(t["y_star"][0]))
        outputs = {"bicubic": bic, "bicubic + BP": backproject(bic, np.asarray(t["y_star"][0]))}
        entry = {}
        for cond in R3_CONDITIONS:
            _, _, a = t[cond]
            outputs[f"CSGM, {cond}"] = a["x_hat"]
            eps_star = np.asarray(t["eps_star"])
            entry[f"eps_err, {cond}"] = float(np.linalg.norm(a["eps"] - eps_star) / np.linalg.norm(eps_star))
        summary["R3"][str(i)] = {"metrics": {n: evaluate(x, t["x_star"], t["y_star"]) for n, x in outputs.items()},
                                 **entry, "loss_at_truth": float(measurement_loss(t["eps_star"], t["y_star"], label,
                                                                                  t["nl_star"]))}
    atomic_write_bytes(path, json.dumps(summary, indent=1).encode())
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
