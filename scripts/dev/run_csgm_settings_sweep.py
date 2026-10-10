"""CSGM settings sweep on 100 development images; it chose the frozen CSGM settings (docs/PROTOCOL.md, Section 3).

Correct-class CSGM on SR x4 over eps, with the generator frozen and z fixed at initialization:

    J(eps) = ||y - A G(eps; z, c)||^2 + w * |y| * R(eps),     |y| = 64*64*3 = 12,288

R = 0 ("none") or the moments penalty R = skew(eps)^2 + 0.25 * excess_kurtosis(eps)^2, with eps standardized over
all 196,608 values.

Grid: Adam lr {0.005, 0.02} x regularizer {none, moments w = 1, 10, 100} = 8 configurations (config_id 0-7),
3,000 steps, 2 restarts each, 1,600 runs in total. Images: one from each of 100 classes (seed 2027), excluding the
classes of the first 4 images of the shuffled development split. Every configuration starts from the same
key(2026, image, restart). Per image the restart with the lower final LR loss is kept, and --analyze ranks the
configurations by mean LR loss. HR PSNR is reported but chose nothing. The winner, config 4 (lr 0.02, no
regularizer), became iadrift.csgm.FROZEN_CONFIG, with the step count raised to 5,000.

The loss and step here are the sweep's own (a generic loss_per_image that takes the operator and regularizer as
arguments). They compile to a different program than iadrift.csgm, so the package version is not used.

Usage:
    python scripts/dev/run_csgm_settings_sweep.py --dry-run     # plan, hash and cache status; runs and writes nothing
    python scripts/dev/run_csgm_settings_sweep.py --analyze     # rank the configurations from the saved results (CPU)
    python scripts/dev/run_csgm_settings_sweep.py --smoke       # rerun 2 saved runs (config 4 and 5), compare exactly
    python scripts/dev/run_csgm_settings_sweep.py --max-runs 32 # run what is missing (resumable)

Outputs (IADRIFT_DATA): phase1_sr4_oracle_moments_dev_v1_6b22675bd557.{jsonl,manifest.json}, one JSON row per run
(settings, initial/final LR loss, penalty and objective, HR PSNR, eps statistics, seconds); no arrays. The 100-image
pool in this file is also where run_csgm_dev.py and run_z_swap_confirm_dev.py draw the development images from.
phase1_sr4_oracle_moments_dev_v1_795139bf9252.* is an abandoned 2,000-step version of the same plan (2 runs).
About 50 s per run on an RTX A4000, so ~22 h for the whole sweep.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

import os

if "--dry-run" in sys.argv or "--analyze" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from iadrift.runtime import setup

setup(theta_gradients=False)  # the sweep ran with JAX's default allocator

import argparse
import json
import math
from collections import defaultdict
from itertools import product
from statistics import fmean
from time import perf_counter

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import Checkpoint, free_gb, load_results

EXPERIMENT_VERSION = "sr4_oracle_moments_dev_v1"
SAVED_HASH = "6b22675bd557"
DEVELOPMENT_SEED = 2027  # the 100 sweep images
NUM_SWEEP_IMAGES = 100
LEARNING_RATES = (0.005, 0.02)
STEP_COUNTS = (3_000,)
RESTART_COUNTS = (2,)
REGULARIZATION_OPTIONS = (
    {"reg_type": "none", "reg_weight": 0.0},
    {"reg_type": "moments", "reg_weight": 1.0},
    {"reg_type": "moments", "reg_weight": 10.0},
    {"reg_type": "moments", "reg_weight": 100.0},
)
SMOKE_KEYS = ((4, 0), (5, 0))  # (config_id, restart) on the first sweep image: no regularizer and moments x1
CONFIG_FIELDS = ("learning_rate", "num_steps", "num_restarts", "reg_type", "reg_weight")


# ---------------------------------------------------------------- plan (no JAX)
def sweep_images(labels, train_indices, test_indices):
    """One development image from each of 100 classes, excluding the classes of train_indices[:4]."""
    rng = np.random.default_rng(DEVELOPMENT_SEED)
    class_ids = np.unique(labels)
    pilot_indices = np.asarray(train_indices[:4], dtype=np.int32)
    eligible = np.setdiff1d(class_ids, np.unique(labels[pilot_indices]))
    selected = rng.choice(eligible, size=NUM_SWEEP_IMAGES, replace=False)
    indices = np.asarray([int(rng.choice(train_indices[labels[train_indices] == c])) for c in selected], dtype=np.int32)
    assert len(np.unique(labels[indices])) == NUM_SWEEP_IMAGES
    assert np.all(np.isin(indices, train_indices)) and not np.any(np.isin(indices, test_indices))
    assert not np.any(np.isin(indices, pilot_indices))
    return indices


def build_run_plan(indices):
    """All configurations of an image (each with its restarts) before the next image."""
    base_grid = [{"learning_rate": lr, "num_steps": steps, "num_restarts": restarts}
                 for lr, steps, restarts in product(LEARNING_RATES, STEP_COUNTS, RESTART_COUNTS)]
    grid = [{**base, **reg} for base in base_grid for reg in REGULARIZATION_OPTIONS]
    return [{"config_id": config_id, "image_index": int(i), "restart": restart, **config}
            for i in indices for config_id, config in enumerate(grid) for restart in range(config["num_restarts"])]


def run_key(run):
    return (int(run["config_id"]), int(run["image_index"]), int(run["restart"]))


# ---------------------------------------------------------------- analysis
def analyze(results_path, manifest_path):
    """Per image and configuration keep the restart with the lowest final LR loss; rank configurations."""
    planned = {run_key(r): r for r in json.loads(manifest_path.read_text())["run_plan"]}
    results = load_results(results_path)
    observed = {}
    for row in results:
        key = run_key(row)
        if key in observed or key not in planned:
            raise ValueError(f"Duplicate or unexpected result: {key}")
        if any(row[f] != planned[key][f] for f in CONFIG_FIELDS):
            raise ValueError(f"Run {key} disagrees with the manifest")
        if not all(isinstance(row[f], (int, float)) and math.isfinite(row[f]) for f in ("final_loss", "psnr")):
            raise ValueError(f"Run {key} has a nonfinite final_loss or psnr")
        observed[key] = row
    if planned.keys() - observed.keys():
        raise ValueError(f"Sweep is incomplete: {len(observed)}/{len(planned)} runs")

    config_by_id, images_by_config = {}, defaultdict(set)
    for row in planned.values():
        config = {f: row[f] for f in CONFIG_FIELDS}
        assert config_by_id.setdefault(row["config_id"], config) == config, "inconsistent configuration"
        images_by_config[row["config_id"]].add(row["image_index"])
    image_sets = list(images_by_config.values())
    assert all(s == image_sets[0] for s in image_sets), "configurations were not run on the same images"

    best = {}
    for row in results:
        key = (row["config_id"], row["image_index"])
        if key not in best or (row["final_loss"], row["restart"]) < (best[key]["final_loss"], best[key]["restart"]):
            best[key] = row
    summary = []
    for config_id in sorted(config_by_id):
        chosen = [best[(config_id, i)] for i in sorted(image_sets[0])]
        summary.append({"config_id": config_id, **config_by_id[config_id], "num_images": len(chosen),
                        "mean_final_loss": fmean(r["final_loss"] for r in chosen),
                        "mean_hr_psnr": fmean(r["psnr"] for r in chosen)})
    by_loss = sorted(summary, key=lambda r: (r["mean_final_loss"], -r["mean_hr_psnr"], r["config_id"]))
    by_psnr = sorted(summary, key=lambda r: (-r["mean_hr_psnr"], r["mean_final_loss"], r["config_id"]))
    return summary, by_loss, by_psnr, len(results)


def print_analysis(ckpt):
    summary, by_loss, by_psnr, num_runs = analyze(ckpt.results_path, ckpt.manifest_path)
    print(f"Validated {num_runs} runs across {summary[0]['num_images']} development images.")
    print("Restart selection: lowest final LR loss per image and configuration. HR PSNR: mean over images (dB).\n")
    print("config  lr      steps  restarts  regularizer  weight   mean loss   mean HR PSNR")
    for r in summary:
        print(f"{r['config_id']:>6}  {r['learning_rate']:<7g} {r['num_steps']:>6}  {r['num_restarts']:>8}  "
              f"{r['reg_type']:<11}  {r['reg_weight']:>6g}  {r['mean_final_loss']:>10.3f}  {r['mean_hr_psnr']:>12.3f}")
    for title, w, metric, unit in (("Lowest loss", by_loss[0], "mean_final_loss", ""),
                                   ("Highest HR PSNR", by_psnr[0], "mean_hr_psnr", " dB")):
        print(f"\n{title}: config {w['config_id']} (lr={w['learning_rate']:g}, steps={w['num_steps']}, "
              f"restarts={w['num_restarts']}, reg={w['reg_type']}, weight={w['reg_weight']:g}); "
              f"{metric}={w[metric]:.3f}{unit}")


# ---------------------------------------------------------------- the sweep's CSGM (GPU)
class Sweep:
    """The sweep's objective, step and run, kept as they were when the sweep ran."""

    def __init__(self, gen, index_rows, lr_all):
        import jax
        import jax.numpy as jnp
        import optax

        from iadrift.operators import A_sr4

        model, params = gen.model, gen.params
        self.index_rows, self.lr_all = index_rows, lr_all

        def eps_regularizer(eps, kind):
            """Return a scalar penalty. `kind` must be fixed when JIT compiling."""
            e = jnp.asarray(eps, dtype=jnp.float32).reshape(-1)
            if kind == "none":
                return jnp.array(0.0, dtype=e.dtype)
            if kind == "l2":
                return 0.5 * jnp.mean(e ** 2)
            mean = jnp.mean(e)
            variance = jnp.mean((e - mean) ** 2)
            z = (e - mean) / jnp.sqrt(variance + 1e-12)
            if kind == "moments":
                skewness = jnp.mean(z ** 3)
                excess_kurtosis = jnp.mean(z ** 4) - 3.0
                return skewness ** 2 + 0.25 * excess_kurtosis ** 2
            if kind == "quantiles":
                n = e.size
                probabilities = (jnp.arange(n, dtype=e.dtype) + 0.5) / n
                normal_quantiles = jax.scipy.special.ndtri(probabilities)
                return jnp.mean((jnp.sort(z) - normal_quantiles) ** 2)
            raise ValueError(f"Unknown regularizer: {kind}")

        def loss_per_image(eps, y, A, label, noise_labels, *, reg_type="none", reg_weight=0.0):
            """Total objective for one image; with the default arguments, the measurement loss."""
            assert eps.shape == (1, 256, 256, 3)
            assert noise_labels.shape == (1, model.noise_coords)
            class_id = jnp.asarray(label, dtype=jnp.int32).reshape(1)
            condition = model.apply({"params": params}, class_id, 1.0, noise_labels, method=model.c_cfg_noise_to_cond)
            x_hat = model.apply({"params": params}, eps, condition, True, method=model.generate_image)
            predicted_y = A(x_hat.astype(jnp.float32))
            measured_y = jnp.asarray(y, dtype=jnp.float32)
            if predicted_y.shape != measured_y.shape:
                raise ValueError(f"Measurement shapes differ: {predicted_y.shape} vs {measured_y.shape}")
            measurement_loss = jnp.sum((predicted_y - measured_y) ** 2)
            penalty = eps_regularizer(eps, reg_type)
            # y.size puts the mean-sized penalty on the scale of the summed measurement loss.
            weighted_penalty = jnp.asarray(reg_weight, dtype=measurement_loss.dtype) * measured_y.size * penalty
            return measurement_loss + weighted_penalty

        def gradient_step(eps, opt_state, y, A, label, noise_labels, optimizer, *, reg_type="none", reg_weight=0.0):
            """One optimizer step; returns the objective before the update."""
            objective, grad_eps = jax.value_and_grad(loss_per_image, argnums=0)(
                eps, y, A, label, noise_labels, reg_type=reg_type, reg_weight=reg_weight)
            updates, new_opt_state = optimizer.update(grad_eps, opt_state, eps)
            return optax.apply_updates(eps, updates), new_opt_state, objective

        def make_train_step(optimizer, reg_type):
            @jax.jit
            def train_step(eps, opt_state, y, label, noise_labels, reg_weight):
                return gradient_step(eps, opt_state, y, A_sr4, label, noise_labels, optimizer,
                                     reg_type=reg_type, reg_weight=reg_weight)
            return train_step

        self.step_cache = {}
        for lr in LEARNING_RATES:
            optimizer = optax.adam(learning_rate=lr)
            for reg_type in sorted({o["reg_type"] for o in REGULARIZATION_OPTIONS}):
                self.step_cache[(lr, reg_type)] = (optimizer, make_train_step(optimizer, reg_type))

        self.jax, self.jnp, self.gen = jax, jnp, gen
        self.A_sr4, self.eps_regularizer, self.loss_per_image = A_sr4, eps_regularizer, loss_per_image
        self.model, self.params = model, params

    def initialize_run(self, run):
        jax, jnp = self.jax, self.jnp
        image_index = int(run["image_index"])
        y = jnp.asarray(np.asarray(self.lr_all[image_index:image_index + 1], dtype=np.float32))
        label = jnp.asarray(int(self.index_rows[image_index]["label"]), dtype=jnp.int32)  # oracle class
        eps, noise_labels = self.gen.initialize(image_index, int(run["restart"]))  # key(2026, image, restart)
        optimizer, train_step = self.step_cache[(run["learning_rate"], run["reg_type"])]
        return y, label, noise_labels, eps, optimizer.init(eps), train_step

    def run_one(self, run):
        jax, jnp, model, params = self.jax, self.jnp, self.model, self.params
        y, label, noise_labels, eps, opt_state, train_step = self.initialize_run(run)
        initial_eps = eps
        initial_loss = float(self.loss_per_image(eps, y, self.A_sr4, label, noise_labels))
        initial_penalty = float(self.eps_regularizer(eps, run["reg_type"]))
        penalty_scale = float(run["reg_weight"]) * y.size
        initial_objective = initial_loss + penalty_scale * initial_penalty

        start_time = perf_counter()
        for step in range(1, run["num_steps"] + 1):
            eps, opt_state, _ = train_step(eps, opt_state, y, label, noise_labels, run["reg_weight"])
            if step % 50 == 0 or step == run["num_steps"]:
                eps.block_until_ready()

        final_loss = float(self.loss_per_image(eps, y, self.A_sr4, label, noise_labels))
        final_penalty = float(self.eps_regularizer(eps, run["reg_type"]))
        condition = model.apply({"params": params}, jnp.asarray(label, dtype=jnp.int32).reshape(1), 1.0,
                                noise_labels, method=model.c_cfg_noise_to_cond)
        x_hat = model.apply({"params": params}, eps, condition, True, method=model.generate_image)
        reconstruction = np.asarray(jax.device_get(x_hat), dtype=np.float32)[0]
        hr = data.hr_image(self.index_rows, run["image_index"])
        return {
            **run,
            "initial_loss": initial_loss, "final_loss": final_loss,
            "initial_penalty": initial_penalty, "final_penalty": final_penalty,
            "initial_objective": initial_objective, "final_objective": final_loss + penalty_scale * final_penalty,
            "psnr": psnr_01(hr, reconstruction),
            "initial_eps_rms": float(jnp.sqrt(jnp.mean(initial_eps ** 2))),
            "final_eps_rms": float(jnp.sqrt(jnp.mean(eps ** 2))),
            "initial_eps_mean": float(jnp.mean(initial_eps)), "final_eps_mean": float(jnp.mean(eps)),
            "initial_eps_std": float(jnp.std(initial_eps)), "final_eps_std": float(jnp.std(eps)),
            "elapsed_seconds": perf_counter() - start_time,
        }


def psnr_01(original, reconstructed, value_range=(-1.0, 1.0)):
    """The sweep's PSNR: both images rescaled to [0, 1] first (same value as iadrift.metrics.psnr up to rounding)."""
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    low, high = value_range
    if original.shape != reconstructed.shape or original.size == 0:
        raise ValueError("Images must have the same nonempty shape.")
    if not np.isfinite(original).all() or not np.isfinite(reconstructed).all():
        raise ValueError("Images must contain only finite values.")
    mse = np.mean(((original - low) / (high - low) - (reconstructed - low) / (high - low)) ** 2)
    return float("inf") if mse == 0 else float(-10 * np.log10(mse))


def smoke(sweep, ckpt, plan):
    saved = {run_key(r): r for r in load_results(ckpt.results_path)}
    first_image = plan[0]["image_index"]
    ok = True
    for config_id, restart in SMOKE_KEYS:
        key = (config_id, first_image, restart)
        row = sweep.run_one(ckpt.plan_by_key[key])
        fields = [k for k in row if k not in ("elapsed_seconds",)]
        diff = [k for k in fields if row[k] != saved[key][k]]
        ok &= not diff
        print(f"[smoke] config {config_id}, image {first_image}, r{restart}: final loss {row['final_loss']!r} vs saved "
              f"{saved[key]['final_loss']!r}; differing fields: {diff or 'none'} ({row['elapsed_seconds']:.0f} s)")
    print("SMOKE PASSED" if ok else "SMOKE FAILED")
    if not ok:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-runs", type=int, default=32, help="runs per call (the sweep is resumable)")
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; run and write nothing")
    parser.add_argument("--analyze", action="store_true", help="rank the configurations from the saved results")
    parser.add_argument("--smoke", action="store_true", help="rerun 2 saved runs and compare exactly; writes nothing")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train_indices, test_indices = data.load_split(labels)
    plan = build_run_plan(sweep_images(labels, train_indices, test_indices))
    spec = {"version": EXPERIMENT_VERSION, "model": "hf://pixel_B_sota", "operator": "A_sr4",
            "sweep_seed": data.SWEEP_SEED, "run_plan": plan}
    ckpt = Checkpoint(paths.DATA_DIR, f"phase1_{EXPERIMENT_VERSION}", spec, run_key)
    if ckpt.hash != SAVED_HASH:
        raise SystemExit(f"Plan hash {ckpt.hash} differs from the saved sweep {SAVED_HASH}.")

    if args.dry_run:
        ckpt.check_manifest()
        done = ckpt.n_done()
        print("DRY RUN: nothing is run or written.")
        print(f"{NUM_SWEEP_IMAGES} development images x 8 configurations (lr {LEARNING_RATES} x regularizer "
              f"{[(o['reg_type'], o['reg_weight']) for o in REGULARIZATION_OPTIONS]}) x 2 restarts, "
              f"{STEP_COUNTS[0]:,} steps")
        print(f"Settings hash: {ckpt.hash} (saved: {SAVED_HASH})   Planned runs: {len(plan)}; already saved: {done}; "
              f"remaining: {len(plan) - done}")
        print(f"Results: {ckpt.results_path} ({free_gb(paths.DATA_DIR):.0f} GB free)")
        return
    if args.analyze:
        print_analysis(ckpt)
        return

    from iadrift.generator import load_generator

    sweep = Sweep(load_generator(), index_rows, data.load_measurements())
    if args.smoke:
        smoke(sweep, ckpt, plan)
        return

    with ckpt.open() as (results, remaining):
        todo = remaining[: args.max_runs]
        print(f"Checkpoint: {ckpt.results_path}\nCompleted: {len(results)} / {len(plan)}; this call: {len(todo)}")
        for run in todo:
            ckpt.append(results, sweep.run_one(run))
            print(f"[{len(results)}/{len(plan)}] image {run['image_index']} config {run['config_id']} "
                  f"r{run['restart']}: loss {results[-1]['final_loss']:.1f}, PSNR {results[-1]['psnr']:.2f} dB",
                  flush=True)
        ckpt.finish(results)


if __name__ == "__main__":
    main()
