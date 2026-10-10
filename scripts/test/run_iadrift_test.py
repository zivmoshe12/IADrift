"""IADrift test run (docs/PROTOCOL.md, Section 4; pre-registered): 100 test images x {correct, wrong label}.

For each image of the fixed 100-image test sample and each label condition:
    start   the saved CSGM test run with the lowest final LR loss (its eps, noise labels z and class label;
            phase1_sr4_csgm_label_test_v2_n100_9e24c78a47ab, scripts/test/run_csgm_test.py)
    theta   the pretrained pixel_B_sota weights (a fresh copy per run)
    loss    ||y - A G_theta(eps; z, c)||^2
    optim   Adam on all 241 theta leaves (lr 1e-5) and Adam on eps (lr 1e-3), one jitted joint step with donated
            buffers (iadrift.adaptation.Adaptation.make_joint_step)
    steps   1,000; outputs at the pre-declared checkpoints 100 and 1,000, each through the eager generator path
One deterministic run per (image, condition); no restarts, nothing chosen with HR (HR is used only for logging PSNR).
Start check: the CSGM output is regenerated eagerly in this process and its max difference from the saved CSGM output
is recorded (0.0 for every saved run). Back-projection, LPIPS, blockiness, band SNR and the primary tests are computed
later by scripts/analysis/evaluate_test.py --part iadrift (notebook dev_archive/03 Part 3).

Ported from scripts/run_sr4_iadrift_test.py (+ generate / iadrift_loss / make_joint_step / eager_metrics of
scripts/run_sr4_iadrift_smoke.py).

Usage:
    python scripts/test/run_iadrift_test.py --dry-run           # plan and cache status; runs and writes nothing (CPU)
    python scripts/test/run_iadrift_test.py --smoke             # 100 steps of saved run 1704/oracle; writes nothing
    python scripts/test/run_iadrift_test.py --max-runs 10       # resumable chunk of 10 runs
    bash scripts/test/run_overnight.sh iadrift                  # all runs, resumable, chunked

Outputs (IADRIFT_DATA): phase2_sr4_iadrift_test_v1_n<N>_<hash>.{jsonl,manifest.json} and ..._arrays/
img#####_{oracle,wrong}.npz (x_step{100,1000}, eps_step{100,1000}, trace_steps, trace_loss, noise_labels). For N = 100
the hash is 00ca59364e18. ~60-70 s per run on an RTX A4000 (200 runs ~3.5 h); runs in chunks of 10 (a 32-run chunk
segfaulted after 16 runs, protocol Section 8).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

if "--dry-run" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the dry run never imports JAX; this is only a safeguard

from iadrift.runtime import setup

setup(theta_gradients=True)  # gradients w.r.t. all 134M weights: XLA_PYTHON_CLIENT_ALLOCATOR=platform

import argparse
import io
import json
from time import perf_counter

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import Checkpoint, atomic_write_bytes, free_gb, load_results

EXPERIMENT_VERSION = "sr4_iadrift_test_v1"
CONDITIONS = ("oracle", "wrong")
CSGM_STEM = "phase1_sr4_csgm_label_test_v2_n100_9e24c78a47ab"
# Kept verbatim (including "protocol" and "start"): every run-plan row carries these keys, so they enter the spec hash.
FROZEN_CONFIG = {
    "protocol": "docs/PROTOCOL.md (frozen 2026-10-05)",
    "lr_theta": 1e-5,
    "lr_eps": 1e-3,
    "num_steps": 1_000,
    "checkpoints": [100, 1_000],
    "cfg": 1.0,
    "start": f"{CSGM_STEM}: lowest final LR loss restart per image and condition",
}
TRACE_EVERY = 50
SMOKE = {"image_index": 1704, "condition": "oracle", "num_steps": 100}  # the disclosed smoke-test image
SAVED_STEM = "phase2_sr4_iadrift_test_v1_n100_00ca59364e18"


def csgm_starts(sample, labels):
    """(image, condition) -> the saved CSGM test run with the lowest final LR loss, with checks."""
    rows = load_results(paths.DATA_DIR / f"{CSGM_STEM}.jsonl")
    plan = json.loads((paths.DATA_DIR / f"{CSGM_STEM}.manifest.json").read_text())["run_plan"]
    keys = {(r["image_index"], r["condition"], r["restart"]) for r in rows}
    assert len(keys) == len(rows) == len(plan), "CSGM test results incomplete or duplicated"
    best = {}
    for r in rows:
        k = (r["image_index"], r["condition"])
        if k not in best or (r["final_loss"], r["restart"]) < (best[k]["final_loss"], best[k]["restart"]):
            best[k] = r
    for i in sample:
        i = int(i)
        true_label = int(labels[i])
        assert best[(i, "oracle")]["class_label"] == true_label
        assert best[(i, "wrong")]["class_label"] == data.wrong_label_for(i, true_label) != true_label
    return best


def build_run_plan(sample, labels, best):
    plan = []
    for i in sample:
        i = int(i)
        for condition in CONDITIONS:
            b = best[(i, condition)]
            plan.append({
                "image_index": i,
                "condition": condition,
                "true_label": int(labels[i]),
                "class_label": int(b["class_label"]),
                "csgm_restart": int(b["restart"]),
                "csgm_array_file": b["array_file"],
                "csgm_final_loss": float(b["final_loss"]),
                **FROZEN_CONFIG,
            })
    return plan


def run_key(run):
    return (int(run["image_index"]), run["condition"])


def iadrift(adapt, joint_step, run, y, hr, num_steps=FROZEN_CONFIG["num_steps"],
            checkpoints=tuple(FROZEN_CONFIG["checkpoints"])):
    """One frozen IADrift run; returns (metrics, arrays). num_steps / checkpoints are shortened only by --smoke.

    The loop of the original runner (not Adaptation.run_iadrift, which also stores x_start and has no start check),
    so the saved .npz files and JSONL rows have exactly the original fields.
    """
    import jax.numpy as jnp
    from iadrift.metrics import psnr

    opt_theta, opt_eps, step_fn = joint_step
    label = run["class_label"]
    with np.load(paths.DATA_DIR / f"{CSGM_STEM}_arrays" / run["csgm_array_file"]) as saved:
        eps = jnp.asarray(saved["eps"])
        noise_labels = jnp.asarray(saved["noise_labels"])
        csgm_x = saved["x_hat"]
    # Start check: the CSGM output regenerated eagerly with the pretrained weights in this process.
    x0, loss0, psnr0 = adapt.eager_metrics(adapt.gen.params, eps, label, noise_labels, y, hr)
    metrics = {"start_lr_loss": loss0, "start_psnr": psnr0,
               "start_max_abs_diff_vs_saved_csgm": float(np.abs(x0 - csgm_x).max())}

    theta = adapt.fresh_weights()  # fresh copy; gen.params itself is never donated
    state_theta, state_eps = opt_theta.init(theta), opt_eps.init(eps)
    arrays, trace_steps, trace_loss = {}, [], []
    start = perf_counter()
    for step in range(1, num_steps + 1):
        theta, eps, state_theta, state_eps, loss_before = step_fn(
            theta, eps, state_theta, state_eps, y, label, noise_labels)
        if step % TRACE_EVERY == 0:
            trace_steps.append(step - 1)  # loss_before is evaluated before update `step`
            trace_loss.append(float(loss_before))
        if step in checkpoints:
            x, loss, p = adapt.eager_metrics(theta, eps, label, noise_labels, y, hr)
            finite = bool(np.all(np.isfinite(x)) and np.isfinite(loss))
            metrics[f"step{step}"] = {"lr_loss": loss, "psnr": p, "psnr_clipped": psnr(hr, np.clip(x, -1, 1)),
                                      "finite": finite}
            arrays[f"x_step{step}"] = np.asarray(x, np.float32)
            arrays[f"eps_step{step}"] = np.asarray(eps, np.float32)
    metrics["elapsed_seconds"] = perf_counter() - start
    arrays["trace_steps"] = np.asarray(trace_steps, np.int32)
    arrays["trace_loss"] = np.asarray(trace_loss, np.float32)
    arrays["noise_labels"] = np.asarray(noise_labels)
    del theta, state_theta, state_eps
    return metrics, arrays


def smoke(adapt, joint_step, plan_by_key, index_rows, lr_all):
    """Rerun the first 100 steps of saved run 1704/oracle; compare with the saved arrays. Writes nothing."""
    import jax.numpy as jnp

    i, condition, n = SMOKE["image_index"], SMOKE["condition"], SMOKE["num_steps"]
    run = plan_by_key[(i, condition)]
    saved_row = next(r for r in load_results(paths.DATA_DIR / f"{SAVED_STEM}.jsonl") if run_key(r) == (i, condition))
    hr = data.hr_image(index_rows, i)
    y = jnp.asarray(lr_all[i:i + 1], jnp.float32)
    m, a = iadrift(adapt, joint_step, run, y, hr, num_steps=n, checkpoints=(n,))
    with np.load(paths.DATA_DIR / f"{SAVED_STEM}_arrays" / saved_row["array_file"]) as saved:
        ref = {"x_step100": saved["x_step100"], "eps_step100": saved["eps_step100"],
               "trace_loss[:2]": saved["trace_loss"][: n // TRACE_EVERY]}
    new = {"x_step100": a["x_step100"], "eps_step100": a["eps_step100"], "trace_loss[:2]": a["trace_loss"]}
    exact = {}
    for k in ref:
        exact[k] = bool(np.array_equal(new[k], ref[k]))
        diff = float(np.abs(new[k].astype(np.float64) - ref[k].astype(np.float64)).max())
        print(f"[smoke] {k}: identical {exact[k]}; max abs diff {diff:.3e}")
    print(f"[smoke] trace_loss[:2] {a['trace_loss'].tolist()} vs saved {ref['trace_loss[:2]'].tolist()}")
    print(f"[smoke] start check: max |regenerated CSGM - saved CSGM| {m['start_max_abs_diff_vs_saved_csgm']:.3e} "
          f"(saved run: {saved_row['start_max_abs_diff_vs_saved_csgm']:.3e}); start LR loss {m['start_lr_loss']!r} "
          f"vs saved {saved_row['start_lr_loss']!r}")
    print(f"[smoke] step-100 LR loss {m['step100']['lr_loss']!r} vs saved {saved_row['step100']['lr_loss']!r}; "
          f"PSNR {m['step100']['psnr']:.4f} vs {saved_row['step100']['psnr']:.4f} ({m['elapsed_seconds']:.0f} s)")
    ok = all(exact.values()) and m["start_max_abs_diff_vs_saved_csgm"] == 0.0
    print("SMOKE PASSED" if ok else "SMOKE FAILED")
    if not ok:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-images", type=int, default=data.TEST_SAMPLE_SIZE,
                        help=f"use the first N images of the fixed {data.TEST_SAMPLE_SIZE}-image test sample")
    parser.add_argument("--max-runs", type=int, default=10, help="runs per call (the run is resumable)")
    parser.add_argument("--dry-run", action="store_true", help="validate the plan and print a summary; run nothing")
    parser.add_argument("--smoke", action="store_true", help="100-step rerun of saved run 1704/oracle; writes nothing")
    args = parser.parse_args()
    if not 1 <= args.num_images <= data.TEST_SAMPLE_SIZE:
        parser.error(f"--num-images must be between 1 and {data.TEST_SAMPLE_SIZE}")

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    lr_all = data.load_measurements()
    _, test_indices = data.load_split(labels)
    sample = data.get_test_sample(labels, test_indices)[: args.num_images]
    best = csgm_starts(sample, labels)
    run_plan = build_run_plan(sample, labels, best)

    spec = {
        "version": EXPERIMENT_VERSION, "model": "hf://pixel_B_sota", "operator": "A_sr4",
        "conditions": list(CONDITIONS), "test_sample_path": paths.TEST_SAMPLE_NPZ,
        "test_sample_seed": data.TEST_SAMPLE_SEED, "wrong_label_seed": data.WRONG_LABEL_SEED,
        "num_images": args.num_images, "run_plan": run_plan,
    }
    ckpt = Checkpoint(paths.DATA_DIR, f"phase2_{EXPERIMENT_VERSION}_n{args.num_images}", spec, run_key)
    assert len(run_plan) == len(sample) * len(CONDITIONS)

    if args.dry_run:
        ckpt.check_manifest()
        done = ckpt.n_done()
        print("DRY RUN: nothing is run or written.")
        print(f"Frozen config: {FROZEN_CONFIG}")
        print(f"Images: {len(sample)} test images (first: {[int(i) for i in sample[:4]]}); conditions {CONDITIONS}")
        first = [(r["image_index"], r["condition"], r["class_label"], f"csgm r{r['csgm_restart']}")
                 for r in run_plan[:4]]
        print(f"First runs: {first}")
        print(f"Settings hash: {ckpt.hash}   Planned runs: {len(run_plan)}; already saved: {done}; "
              f"remaining: {len(run_plan) - done}")
        n_arrays = sum((ckpt.arrays_dir / f"img{r['image_index']:05d}_{r['condition']}.npz").exists() for r in run_plan)
        print(f"Results:  {ckpt.results_path}\nManifest: {ckpt.manifest_path} "
              f"({'exists' if ckpt.manifest_path.exists() else 'will be created'})")
        print(f"Arrays:   {ckpt.arrays_dir}/ ({n_arrays} / {len(run_plan)} present; ~3 MB per run; "
              f"{free_gb(paths.DATA_DIR):.0f} GB free)")
        print(f"Estimated time at ~65 s/run: {(len(run_plan) - done) * 65 / 3600:.1f} h")
        return

    import jax.numpy as jnp
    from iadrift.adaptation import Adaptation
    from iadrift.generator import load_generator

    adapt = Adaptation(load_generator())
    joint_step = adapt.make_joint_step(FROZEN_CONFIG["lr_theta"], FROZEN_CONFIG["lr_eps"])
    if args.smoke:
        smoke(adapt, joint_step, ckpt.plan_by_key, index_rows, lr_all)
        return

    with ckpt.open() as (results, remaining):
        todo = remaining[: args.max_runs]
        print(f"Checkpoint: {ckpt.results_path}\nCompleted: {len(results)} / {len(run_plan)}; this call: {len(todo)}",
              flush=True)
        for run in todo:
            i, condition = run["image_index"], run["condition"]
            hr = data.hr_image(index_rows, i)
            y = jnp.asarray(lr_all[i:i + 1], jnp.float32)
            metrics, arrays = iadrift(adapt, joint_step, run, y, hr)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            array_path = ckpt.arrays_dir / f"img{i:05d}_{condition}.npz"
            atomic_write_bytes(array_path, buffer.getvalue())
            ckpt.append(results, {**run, **metrics, "array_file": array_path.name})
            s100, s1000 = metrics["step100"], metrics["step1000"]
            print(f"[{len(results)}/{len(run_plan)}] image {i} {condition:<6} (class {run['class_label']}): "
                  f"LR loss {metrics['start_lr_loss']:.1f} -> {s100['lr_loss']:.1f} (100) -> {s1000['lr_loss']:.1f} "
                  f"(1000); PSNR {metrics['start_psnr']:.2f} -> {s100['psnr']:.2f} -> {s1000['psnr']:.2f} dB; "
                  f"start diff {metrics['start_max_abs_diff_vs_saved_csgm']:.1e}; "
                  f"{'' if s100['finite'] and s1000['finite'] else 'NON-FINITE; '}{metrics['elapsed_seconds']:.0f}s",
                  flush=True)
        ckpt.finish(results)


if __name__ == "__main__":
    main()
