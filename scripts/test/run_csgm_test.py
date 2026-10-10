"""CSGM test run (docs/PROTOCOL.md, Section 3): 100 test images x {correct, wrong label} x 2 restarts.

Uses the frozen CSGM settings in iadrift.csgm.FROZEN_CONFIG: Adam, lr 0.02, 5,000 steps, 2 restarts, no regularizer.
These are the best settings from the development sweep (scripts/dev/run_csgm_settings_sweep.py), with the step count
raised from 3,000 to 5,000.

Each image is run under two label conditions with the same (eps, z) initialization: "oracle" uses the true class,
"wrong" uses one other class drawn per image with WRONG_LABEL_SEED. The restart with the lowest final LR loss is
picked later, in scripts/analysis/evaluate_test.py. The HR image is only used to log PSNR.

Usage:
    python scripts/test/run_csgm_test.py --dry-run               # validate the plan; run and write nothing
    python scripts/test/run_csgm_test.py --repro-check           # rerun saved run 1704/oracle/r0, require bit-exact
    python scripts/test/run_csgm_test.py --num-images 1          # smoke test: the first image of the sample
    python scripts/test/run_csgm_test.py --max-runs 10           # resumable; or: bash scripts/test/run_overnight.sh csgm

Outputs (in IADRIFT_DATA):
    phase1_sr4_csgm_label_test_v2_n<N>_<hash>.{jsonl,manifest.json}
    phase1_sr4_csgm_label_test_v2_n<N>_<hash>_arrays/img#####_{oracle,wrong}_r{0,1}.npz
        eps, noise_labels, x_hat, loss trace
For N = 100 the hash is 9e24c78a47ab. About 65 s per run on an RTX A4000, so 400 runs take ~7.5 h.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `iadrift` imports without install

from iadrift.runtime import setup

setup(theta_gradients=False)  # saved CSGM runs reproduce bit-exactly only with the default allocator

import argparse
import io

import numpy as np
from tqdm.auto import tqdm

from iadrift import data, paths
from iadrift.checkpoint import Checkpoint, atomic_write_bytes, free_gb, load_results
from iadrift.csgm import CSGM, FROZEN_CONFIG
from iadrift.generator import load_generator
from iadrift.metrics import psnr
from iadrift.operators import bicubic_up

EXPERIMENT_VERSION = "sr4_csgm_label_test_v2"  # v2: 5,000 steps (v1: 3,000)
CONDITIONS = ("oracle", "wrong")
REPRO = {"image_index": 1704, "condition": "oracle", "restart": 0}  # the disclosed smoke-test image
SAVED_STEM = "phase1_sr4_csgm_label_test_v2_n100_9e24c78a47ab"


def build_run_plan(sample, labels):
    """All runs, image by image: oracle r0, r1, then wrong r0, r1."""
    plan = []
    for image_index in sample:
        image_index = int(image_index)
        true_label = int(labels[image_index])
        class_label = {"oracle": true_label, "wrong": data.wrong_label_for(image_index, true_label)}
        for condition in CONDITIONS:
            for restart in range(FROZEN_CONFIG["num_restarts"]):
                plan.append({"image_index": image_index, "condition": condition, "restart": restart,
                             "true_label": true_label, "class_label": class_label[condition], **FROZEN_CONFIG})
    return plan


def run_key(run):
    return (int(run["image_index"]), run["condition"], int(run["restart"]))


def repro_check(csgm, index_rows, labels, lr_all):
    """Rerun one saved test run and require identical final loss, eps, output and loss trace. Writes nothing."""
    i, condition, restart = REPRO["image_index"], REPRO["condition"], REPRO["restart"]
    saved_row = next(r for r in load_results(paths.DATA_DIR / f"{SAVED_STEM}.jsonl")
                     if (r["image_index"], r["condition"], r["restart"]) == (i, condition, restart))
    hr = data.hr_image(index_rows, i)
    m, a = csgm.reconstruct(i, restart, int(labels[i]), lr_all[i], hr)
    with np.load(paths.DATA_DIR / f"{SAVED_STEM}_arrays" / saved_row["array_file"]) as saved:
        same = {k: bool(np.array_equal(a[k], saved[k])) for k in ("eps", "noise_labels", "x_hat", "trace_loss")}
    exact = m["final_loss"] == saved_row["final_loss"] and all(same.values())
    print(f"[repro] test image {i} {condition} r{restart}: final loss {m['final_loss']!r} vs saved "
          f"{saved_row['final_loss']!r}; identical arrays {same} ({m['elapsed_seconds']:.0f} s)")
    print("REPRO CHECK PASSED" if exact else "REPRO CHECK FAILED")
    if not exact:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-images", type=int, default=data.TEST_SAMPLE_SIZE,
                        help=f"use the first N images of the fixed {data.TEST_SAMPLE_SIZE}-image test sample")
    parser.add_argument("--max-runs", type=int, default=32, help="runs per call (the run is resumable)")
    parser.add_argument("--dry-run", action="store_true", help="validate the plan and print a summary; run nothing")
    parser.add_argument("--repro-check", action="store_true", help="bit-exact rerun of a saved run; writes nothing")
    args = parser.parse_args()
    if not 1 <= args.num_images <= data.TEST_SAMPLE_SIZE:
        parser.error(f"--num-images must be between 1 and {data.TEST_SAMPLE_SIZE}")

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    lr_all = data.load_measurements()
    _, test_indices = data.load_split(labels)
    sample = data.get_test_sample(labels, test_indices)[: args.num_images]
    run_plan = build_run_plan(sample, labels)

    # The spec hash names the output files, so runs with different settings never mix.
    spec = {
        "version": EXPERIMENT_VERSION, "model": "hf://pixel_B_sota", "operator": "A_sr4",
        "conditions": list(CONDITIONS), "sweep_seed": data.SWEEP_SEED, "test_sample_path": paths.TEST_SAMPLE_NPZ,
        "test_sample_seed": data.TEST_SAMPLE_SEED, "wrong_label_seed": data.WRONG_LABEL_SEED,
        "num_images": args.num_images, "run_plan": run_plan,
    }
    ckpt = Checkpoint(paths.DATA_DIR, f"phase1_{EXPERIMENT_VERSION}_n{args.num_images}", spec, run_key)
    assert len(run_plan) == len(sample) * len(CONDITIONS) * FROZEN_CONFIG["num_restarts"]

    if args.dry_run:
        ckpt.check_manifest()
        for run in run_plan:
            assert (run["class_label"] == run["true_label"]) == (run["condition"] == "oracle"), run
        done = ckpt.n_done()
        print("DRY RUN: nothing is run or written.")
        print(f"Images: {len(sample)} test images; conditions {CONDITIONS}; restarts {FROZEN_CONFIG['num_restarts']}; "
              f"steps {FROZEN_CONFIG['num_steps']:,}; lr {FROZEN_CONFIG['learning_rate']}")
        print(f"Settings hash: {ckpt.hash}   Planned runs: {len(run_plan)}; already saved: {done}; "
              f"remaining: {len(run_plan) - done}")
        print(f"Results: {ckpt.results_path}\nArrays:  {ckpt.arrays_dir}/ (~1.6 MB per run; "
              f"{free_gb(paths.DATA_DIR):.0f} GB free)\nEstimated time at ~65 s/run: {(len(run_plan) - done) * 65 / 3600:.1f} h")
        return

    csgm = CSGM(load_generator())
    if args.repro_check:
        repro_check(csgm, index_rows, labels, lr_all)
        return

    with ckpt.open() as (results, remaining):
        todo = remaining[: args.max_runs]
        print(f"Checkpoint: {ckpt.results_path}\nCompleted: {len(results)} / {len(run_plan)}; this call: {len(todo)}")
        for run in tqdm(todo, desc="CSGM test", unit="run", mininterval=10):
            i, condition, restart = run["image_index"], run["condition"], run["restart"]
            hr = data.hr_image(index_rows, i)
            metrics, arrays = csgm.reconstruct(i, restart, run["class_label"], lr_all[i], hr)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            array_path = ckpt.arrays_dir / f"img{i:05d}_{condition}_r{restart}.npz"
            atomic_write_bytes(array_path, buffer.getvalue())
            row = {**run, **metrics, "bicubic_psnr": psnr(hr, bicubic_up(lr_all[i])), "array_file": array_path.name}
            ckpt.append(results, row)
            tqdm.write(f"[{len(results)}/{len(run_plan)}] image {i} {condition:<6} r{restart} (class {run['class_label']}): "
                       f"loss {metrics['final_loss']:.1f}, PSNR {metrics['psnr']:.2f} dB "
                       f"(bicubic {row['bicubic_psnr']:.2f}), {metrics['elapsed_seconds']:.0f}s")
        ckpt.finish(results)


if __name__ == "__main__":
    main()
