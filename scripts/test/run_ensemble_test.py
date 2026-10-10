"""Shifted CSGM ensemble members on the test set (docs/PROTOCOL.md, Section 11, Addendum A; post-hoc, 100 images).

For each test image, correct label only, three new frozen-CSGM runs (iadrift.csgm.FROZEN_CONFIG: Adam lr 0.02, 5,000
steps, no regularizer) on the LR measurement circularly shifted by (sy, sx) LR pixels:
    y_s = roll(y, (sy, sx)),   (sy, sx) in {(0, 1), (1, 0), (1, 1)}
All three start from initialize(image, 0), the same eps and z as the saved test restart r0. The saved x_hat is in the
shifted frame; the analysis shifts it back by -4 (sy, sx) HR pixels (iadrift.ensemble.unshift_x). The two unshifted
members are the saved CSGM test restarts r0 and r1. The ensembles (E1, E2, Ens-U, Ens-W, each + BP) and the two
primary tests are computed by scripts/analysis/evaluate_test.py --part ensemble. The method is the same as in
development Check 13. Nothing is chosen with HR.

Usage:
    python scripts/test/run_ensemble_test.py --dry-run          # plan and cache status; runs and writes nothing (CPU)
    python scripts/test/run_ensemble_test.py --smoke            # two bit-exact 5,000-step reruns; writes nothing
    python scripts/test/run_ensemble_test.py --max-runs 10      # resumable chunk of 10 runs
    bash scripts/test/run_overnight.sh ensemble                 # all runs, resumable, chunked

--smoke (alias --repro-check) reruns saved CSGM test run 1704/oracle/r0 (unshifted) and development image 9780 at
shift (0, 1) from the Check 13 cache ensemble_dev_v1/. Both must match the saved final loss, eps, noise labels, x_hat
and loss trace exactly. Takes about 2 x 65 s.

Outputs (in IADRIFT_DATA/csgm_test_ensemble_v1/, or ..._n<N>/ for N < 100):
    manifest.json                    {"hash", **spec}
    results_<hash>.jsonl             one row per run
    img#####_shift{sy}{sx}_r0.npz    metrics_json, eps, noise_labels, x_hat (shifted frame), trace_steps, trace_loss
For N = 100 the hash is ca2d0ec5e6ff. About 65 s per run (300 runs, ~5.5 h).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `iadrift` imports without install

if "--dry-run" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the dry run never imports JAX; just in case

from iadrift.runtime import setup

setup(theta_gradients=False)  # saved CSGM runs reproduce bit-exactly only with the default allocator

import argparse
import fcntl
import json

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import atomic_save_results, atomic_write_bytes, load_npz, load_results, save_npz, spec_hash
from iadrift.csgm import FROZEN_CONFIG
from iadrift.ensemble import SHIFTS, START_RESTART, shift_y, unshift_x

EXPERIMENT_VERSION = "sr4_csgm_ensemble_test_v1"
OUT_DIR = paths.DATA_DIR / "csgm_test_ensemble_v1"
CSGM_TEST_STEM = "phase1_sr4_csgm_label_test_v2_n100_9e24c78a47ab"
DEV_ENSEMBLE_DIR = paths.DATA_DIR / "ensemble_dev_v1"


def run_member(csgm, image_index, sy, sx, label, y_lr, hr):
    """One shifted CSGM run. Returns (metrics, arrays); arrays['x_hat'] is in the shifted frame."""
    from iadrift.metrics import psnr
    from iadrift.operators import lr_loss

    metrics, arrays = csgm.reconstruct(image_index, START_RESTART, label, shift_y(y_lr, sy, sx), hr)
    x_back = unshift_x(arrays["x_hat"], sy, sx)
    metrics = {
        "final_loss": metrics["final_loss"],          # data term on the shifted y
        "initial_loss": metrics["initial_loss"],
        "lr_loss_unshifted": lr_loss(x_back, y_lr),   # member's loss in original coordinates
        "psnr_unshifted": psnr(hr, x_back),
        "final_eps_rms": metrics["final_eps_rms"],
        "seconds": metrics["elapsed_seconds"],
    }
    return metrics, arrays


def identical(a, b, keys=("eps", "noise_labels", "x_hat", "trace_loss")):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in keys)


def smoke(csgm, index_rows, labels, lr_all):
    """Two bit-exact reproductions; writes nothing. Exits non-zero on mismatch."""
    ok = True

    # 1. Saved CSGM test run: image 1704, oracle, restart 0, unshifted.
    i = 1704
    hr = data.hr_image(index_rows, i)
    m, a = csgm.reconstruct(i, 0, int(labels[i]), lr_all[i], hr)
    saved_row = next(r for r in load_results(paths.DATA_DIR / f"{CSGM_TEST_STEM}.jsonl")
                     if r["image_index"] == i and r["condition"] == "oracle" and r["restart"] == 0)
    with np.load(paths.DATA_DIR / f"{CSGM_TEST_STEM}_arrays" / f"img{i:05d}_oracle_r0.npz") as saved:
        exact = m["final_loss"] == saved_row["final_loss"] and identical(a, saved)
    print(f"[smoke 1] test image {i} oracle r0 unshifted: final loss {m['final_loss']!r} "
          f"vs saved {saved_row['final_loss']!r}; eps/x_hat/trace identical: {exact} ({m['elapsed_seconds']:.0f}s)")
    ok &= exact

    # 2. Shift path: development image 9780, shift (0, 1), vs the Check 13 cache.
    i, (sy, sx) = 9780, (0, 1)
    hr = data.hr_image(index_rows, i)
    m, a = run_member(csgm, i, sy, sx, int(labels[i]), lr_all[i], hr)
    saved_m, saved = load_npz(DEV_ENSEMBLE_DIR / f"img{i:05d}_shift{sy}{sx}_r0.npz")
    exact = m["final_loss"] == saved_m["final_loss"] and identical(a, saved)
    print(f"[smoke 2] dev image {i} shift ({sy}, {sx}): final loss {m['final_loss']!r} "
          f"vs saved {saved_m['final_loss']!r}; eps/x_hat/trace identical: {exact} ({m['seconds']:.0f}s)")
    ok &= exact

    print("SMOKE PASSED" if ok else "SMOKE FAILED")
    if not ok:
        raise SystemExit(1)


def build_run_plan(sample, labels):
    return [{"image_index": int(i), "shift": [sy, sx], "start_restart": START_RESTART,
             "class_label": int(labels[int(i)]), "condition": "oracle", **FROZEN_CONFIG}
            for i in sample for sy, sx in SHIFTS]


def run_key(run):
    return (int(run["image_index"]), *map(int, run["shift"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-images", type=int, default=data.TEST_SAMPLE_SIZE,
                        help=f"use the first N images of the fixed {data.TEST_SAMPLE_SIZE}-image test sample")
    parser.add_argument("--max-runs", type=int, default=10, help="runs per call (the run is resumable)")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan; run and write nothing")
    parser.add_argument("--smoke", "--repro-check", dest="smoke", action="store_true",
                        help="bit-exact reproduction checks; write nothing")
    args = parser.parse_args()
    if not 1 <= args.num_images <= data.TEST_SAMPLE_SIZE:
        parser.error(f"--num-images must be between 1 and {data.TEST_SAMPLE_SIZE}")

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    lr_all = data.load_measurements()

    if args.smoke:
        from iadrift.csgm import CSGM
        from iadrift.generator import load_generator
        smoke(CSGM(load_generator()), index_rows, labels, lr_all)
        return

    _, test_indices = data.load_split(labels)
    sample = data.get_test_sample(labels, test_indices)[: args.num_images]
    run_plan = build_run_plan(sample, labels)
    plan_by_key = {run_key(r): r for r in run_plan}
    assert len(plan_by_key) == len(run_plan) == len(sample) * len(SHIFTS)

    spec = {
        "version": EXPERIMENT_VERSION,
        "model": "hf://pixel_B_sota",
        "operator": "A_sr4",
        "shifts": [list(s) for s in SHIFTS],
        "shift_mode": "circular roll of y by (sy, sx) LR pixels; output rolled back by -4*(sy, sx)",
        "sweep_seed": data.SWEEP_SEED,
        "test_sample_path": paths.TEST_SAMPLE_NPZ,
        "num_images": args.num_images,
        "unshifted_members": f"{CSGM_TEST_STEM}_arrays/img#####_oracle_r{{0,1}}.npz",
        "run_plan": run_plan,
    }
    _, plan_hash = spec_hash(spec)
    out_dir = (OUT_DIR if args.num_images == data.TEST_SAMPLE_SIZE
               else OUT_DIR.with_name(f"{OUT_DIR.name}_n{args.num_images}"))
    manifest_path = out_dir / "manifest.json"
    results_path = out_dir / f"results_{plan_hash}.jsonl"

    if args.dry_run:
        done = len(load_results(results_path)) if results_path.exists() else 0
        if manifest_path.exists() and json.loads(manifest_path.read_text())["hash"] != plan_hash:
            raise RuntimeError(f"Existing manifest does not match this plan: {manifest_path}")
        n_arrays = sum((out_dir / f"img{i:05d}_shift{sy}{sx}_r0.npz").exists() for i, sy, sx in plan_by_key)
        print("DRY RUN: nothing is run or written.")
        print(f"Images: {len(sample)} test images (first {list(map(int, sample[:4]))}), oracle label, "
              f"shifts {SHIFTS}, start restart {START_RESTART}; steps {FROZEN_CONFIG['num_steps']:,}, "
              f"lr {FROZEN_CONFIG['learning_rate']}")
        print(f"Settings hash: {plan_hash}   Planned runs: {len(run_plan)}; already saved: {done}; "
              f"remaining: {len(run_plan) - done}")
        print(f"Output: {out_dir}/ (manifest {'exists' if manifest_path.exists() else 'will be created'}; "
              f"{n_arrays} / {len(run_plan)} member .npz present)")
        print(f"Estimated time at ~65 s/run: {(len(run_plan) - done) * 65 / 3600:.1f} h")
        return

    from iadrift.csgm import CSGM
    from iadrift.generator import load_generator

    csgm = CSGM(load_generator())
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / ".lock").open("a+") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Checkpoint is locked by another process.") from error

        # Own layout (hash inside manifest.json, results_<hash>.jsonl) rather than iadrift.checkpoint's, so the
        # existing cache is still recognised.
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())["hash"] != plan_hash:
                raise RuntimeError("Manifest does not match the current run plan.")
        else:
            if any(out_dir.glob("results_*.jsonl")):
                raise RuntimeError("Results exist without a manifest; refusing to mix runs.")
            atomic_write_bytes(manifest_path, (json.dumps({"hash": plan_hash, **spec}, sort_keys=True) + "\n").encode())

        results = load_results(results_path) if results_path.exists() else []
        completed = {run_key(r) for r in results}
        if len(completed) != len(results) or not completed <= plan_by_key.keys():
            raise RuntimeError("Checkpoint has duplicate or unexpected runs.")
        todo = [r for r in run_plan if run_key(r) not in completed][: args.max_runs]
        print(f"Checkpoint: {results_path}\nCompleted: {len(completed)} / {len(run_plan)}; this call: {len(todo)}",
              flush=True)

        for run in todo:
            i, (sy, sx) = run["image_index"], run["shift"]
            hr = data.hr_image(index_rows, i)
            metrics, arrays = run_member(csgm, i, sy, sx, run["class_label"], lr_all[i], hr)
            if not (np.isfinite(metrics["final_loss"]) and np.all(np.isfinite(arrays["x_hat"]))):
                raise RuntimeError(f"Non-finite result for image {i} shift ({sy}, {sx}); rerun identically.")
            array_path = out_dir / f"img{i:05d}_shift{sy}{sx}_r0.npz"
            save_npz(array_path, arrays, metrics)
            row = {**run, **metrics, "array_file": array_path.name}
            atomic_save_results(results_path, results + [row])
            results.append(row)
            print(f"[{len(results)}/{len(run_plan)}] image {i} shift ({sy}, {sx}): loss {metrics['final_loss']:.1f} "
                  f"(unshifted {metrics['lr_loss_unshifted']:.1f}), PSNR {metrics['psnr_unshifted']:.2f} dB, "
                  f"{metrics['seconds']:.0f}s", flush=True)

        print(f"Saved {len(results)} / {len(run_plan)} results to {results_path}")
        if len(results) == len(run_plan):
            print(f"ALL RUNS COMPLETE: {results_path}")


if __name__ == "__main__":
    main()
