"""Frozen CSGM-like SR x4 benchmark on held-out test images: oracle vs wrong label.

Method: development-sweep config 4 (Adam lr 0.02, 2 restarts, no eps
regularizer), with optimizer steps raised from the swept 3,000 to 5,000.

Conditions per test image, with identical eps/noise_labels initialization:
    oracle  the true ImageNet class
    wrong   one random other class (fixed per image by WRONG_LABEL_SEED)

Test sample: 100 test images from 100 distinct classes, drawn once with
TEST_SAMPLE_SEED and saved to TEST_SAMPLE_PATH. --num-images N uses the
first N images of that fixed sample, so a 1-image smoke test runs exactly the
first image of the full benchmark (under its own checkpoint files).

Usage:
    # Reproducibility check: rerun one saved dev run, compare with the sweep.
    python scripts/run_sr4_csgm_test.py --check-dev

    # Smoke test: 1 image x 2 conditions x 2 restarts = 4 runs.
    python scripts/run_sr4_csgm_test.py --num-images 1

    # Full benchmark, resumable, at most --max-runs runs per invocation.
    python scripts/run_sr4_csgm_test.py --num-images 100 --max-runs 32

Each completed run writes an .npz (eps, noise_labels, x_hat, loss trace)
and then the JSONL checkpoint, both atomically. When every planned run is
saved, the script prints a line starting with ALL RUNS COMPLETE.
"""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

UPSTREAM_DIR = Path.home() / "work" / "drifting"
CACHE_DIR = Path.home() / "work" / "hf_cache"
DATA_DIR = Path.home() / "work" / "datasets" / "ImageNetV2"

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["HF_ROOT"] = str(CACHE_DIR)
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from utils.init_util import load_generator_model_and_params


EXPERIMENT_VERSION = "sr4_csgm_label_test_v2"  # v2: 5,000 steps (v1: 3,000)
FROZEN_CONFIG = {
    "dev_config_id": 4,
    "learning_rate": 0.02,
    "num_steps": 5_000,  # dev sweep tested only 3,000
    "num_restarts": 2,
    "reg_type": "none",
    "reg_weight": 0.0,
}
CONDITIONS = ("oracle", "wrong")
# Same initialization scheme as the dev sweep: key(SWEEP_SEED, image, restart).
SWEEP_SEED = 2026
TEST_SAMPLE_SEED = 2028
TEST_SAMPLE_SIZE = 100
WRONG_LABEL_SEED = 2029
NUM_CLASSES = 1_000
TRACE_EVERY = 50

SPLIT_PATH = DATA_DIR / "imageNetV2_split_80_20_seed2026.npz"
TEST_SAMPLE_PATH = DATA_DIR / f"sr4_test_sample_{TEST_SAMPLE_SIZE}_seed{TEST_SAMPLE_SEED}.npz"
DEV_SWEEP_PATH = DATA_DIR / "phase1_sr4_oracle_moments_dev_v1_6b22675bd557.jsonl"


# ---------------------------------------------------------------- data
def load_hr256(path):
    """Load an image as a centered 256x256 RGB array in [-1, 1]."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    if width <= height:
        new_size = (256, round(height * 256 / width))
    else:
        new_size = (round(width * 256 / height), 256)
    image = image.resize(new_size, Image.Resampling.BICUBIC)
    left = (image.width - 256) // 2
    top = (image.height - 256) // 2
    image = image.crop((left, top, left + 256, top + 256))
    return np.asarray(image, dtype=np.float32) / 127.5 - 1.0


def A_sr4(x):
    """Antialiased bicubic: [batch, 256, 256, 3] -> [batch, 64, 64, 3]."""
    assert x.ndim == 4 and x.shape[1:] == (256, 256, 3)
    return jax.image.resize(
        x.astype(jnp.float32), (x.shape[0], 64, 64, 3),
        method="cubic", antialias=True,
    )


def bicubic_up(y):
    """[64, 64, 3] -> [256, 256, 3], the bicubic baseline."""
    return np.asarray(jax.image.resize(
        jnp.asarray(y, dtype=jnp.float32), (256, 256, 3),
        method="cubic", antialias=False,
    ))


def psnr(original, reconstructed):
    """PSNR for images in [-1, 1] (data range 2)."""
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    assert original.shape == reconstructed.shape
    mse = np.mean(((original - reconstructed) / 2.0) ** 2)
    return float("inf") if mse == 0 else float(-10 * np.log10(mse))


def load_index():
    with (DATA_DIR / "imageNet_x4_index.csv").open(newline="") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: int(row["index"]))
    assert [int(row["index"]) for row in rows] == list(range(10_000))
    return rows


def load_split(labels):
    """Read (never rewrite) the 80/20 split and check its invariants."""
    split = np.load(SPLIT_PATH)
    train, test = split["train_indices"], split["test_indices"]
    assert int(split["seed"]) == 2026
    assert len(train) == 8_000 and len(test) == 2_000
    assert np.intersect1d(train, test).size == 0
    assert np.all(np.bincount(labels[test], minlength=NUM_CLASSES) == 2)
    return train, test


def get_test_sample(labels, test_indices):
    """One test image from each of TEST_SAMPLE_SIZE distinct classes.

    Created once, then always read back, so every method uses the same file.
    """
    rng = np.random.default_rng(TEST_SAMPLE_SEED)
    classes = rng.choice(np.unique(labels[test_indices]), size=TEST_SAMPLE_SIZE, replace=False)
    sample = np.asarray(
        [int(rng.choice(test_indices[labels[test_indices] == c])) for c in classes],
        dtype=np.int32,
    )
    assert len(np.unique(labels[sample])) == TEST_SAMPLE_SIZE
    assert np.all(np.isin(sample, test_indices))

    if TEST_SAMPLE_PATH.exists():
        saved = np.load(TEST_SAMPLE_PATH)["test_sample_indices"]
        if not np.array_equal(saved, sample):
            raise RuntimeError(f"{TEST_SAMPLE_PATH} differs from the seeded draw.")
    else:
        buffer = io.BytesIO()
        np.savez(buffer, test_sample_indices=sample, seed=TEST_SAMPLE_SEED,
                 split_path=str(SPLIT_PATH))
        atomic_write_bytes(TEST_SAMPLE_PATH, buffer.getvalue())
        print(f"Saved test sample: {TEST_SAMPLE_PATH}")
    return sample


def wrong_label_for(image_index, true_label):
    """A uniformly random class != true_label, fixed per image.

    Seeded by (WRONG_LABEL_SEED, image_index), so it does not depend on
    --num-images or on the order in which images are processed.
    """
    rng = np.random.default_rng([WRONG_LABEL_SEED, image_index])
    offset = int(rng.integers(1, NUM_CLASSES))  # 1..999, never 0
    return (true_label + offset) % NUM_CLASSES


# ---------------------------------------------------------------- IO
def atomic_write_bytes(path, data):
    """Write a complete temporary file, then atomically replace path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_save_results(path, rows):
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    atomic_write_bytes(path, payload)


def load_results(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def run_key(run):
    return (int(run["image_index"]), run["condition"], int(run["restart"]))


# ---------------------------------------------------------------- model
model, params, _ = load_generator_model_and_params("hf://pixel_B_sota", hf_cache_dir=str(CACHE_DIR))
assert jax.default_backend() == "gpu", "Refusing to run the benchmark on CPU."


def generate(eps, label, noise_labels):
    condition = model.apply(
        {"params": params}, jnp.asarray(label, dtype=jnp.int32).reshape(1),
        1.0, noise_labels, method=model.c_cfg_noise_to_cond,
    )
    return model.apply({"params": params}, eps, condition, True, method=model.generate_image)


def measurement_loss(eps, y, label, noise_labels):
    x_hat = generate(eps, label, noise_labels)
    return jnp.sum((A_sr4(x_hat.astype(jnp.float32)) - y) ** 2)


optimizer = optax.adam(learning_rate=FROZEN_CONFIG["learning_rate"])


@jax.jit
def train_step(eps, opt_state, y, label, noise_labels):
    loss, grad = jax.value_and_grad(measurement_loss)(eps, y, label, noise_labels)
    updates, opt_state = optimizer.update(grad, opt_state, eps)
    return optax.apply_updates(eps, updates), opt_state, loss


def initialize(image_index, restart):
    """Identical to the dev sweep's initialize_run for the same image/restart.

    Deliberately independent of the class label: oracle and wrong-label runs
    of the same image/restart start from the same eps and noise_labels.
    """
    key = jax.random.PRNGKey(SWEEP_SEED)
    key = jax.random.fold_in(key, image_index)
    key = jax.random.fold_in(key, restart)
    key_eps, key_style = jax.random.split(key)
    eps = jax.random.normal(key_eps, (1, 256, 256, 3), dtype=jnp.float32)
    noise_labels = jax.random.randint(
        key_style, shape=(1, model.noise_coords),
        minval=0, maxval=model.noise_classes, dtype=jnp.int32,
    )
    return eps, noise_labels


def reconstruct(image_index, restart, label, y_lr, hr, num_steps=FROZEN_CONFIG["num_steps"]):
    """Run frozen CSGM for one image/restart/label; return (metrics, arrays)."""
    y = jnp.asarray(y_lr[None], dtype=jnp.float32)
    eps, noise_labels = initialize(image_index, restart)
    opt_state = optimizer.init(eps)
    initial_loss = float(measurement_loss(eps, y, label, noise_labels))

    trace_steps, trace_loss = [], []
    start = perf_counter()
    for step in range(1, num_steps + 1):
        eps, opt_state, loss_before = train_step(eps, opt_state, y, label, noise_labels)
        if step % TRACE_EVERY == 0:
            trace_steps.append(step - 1)  # loss_before is evaluated before update `step`
            trace_loss.append(float(loss_before))
    eps.block_until_ready()
    elapsed = perf_counter() - start

    final_loss = float(measurement_loss(eps, y, label, noise_labels))
    x_hat = np.asarray(jax.device_get(generate(eps, label, noise_labels)), dtype=np.float32)[0]

    metrics = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "psnr": psnr(hr, x_hat),
        "psnr_clipped": psnr(hr, np.clip(x_hat, -1.0, 1.0)),
        "final_eps_rms": float(jnp.sqrt(jnp.mean(eps ** 2))),
        "final_eps_mean": float(jnp.mean(eps)),
        "final_eps_std": float(jnp.std(eps)),
        "elapsed_seconds": elapsed,
    }
    arrays = {
        "eps": np.asarray(eps, dtype=np.float32),
        "noise_labels": np.asarray(noise_labels),
        "x_hat": x_hat,
        "trace_steps": np.asarray(trace_steps, dtype=np.int32),
        "trace_loss": np.asarray(trace_loss, dtype=np.float32),
    }
    return metrics, arrays


# ---------------------------------------------------------------- main
def check_dev(index_rows, lr_all):
    """Rerun the first saved config-4 dev row and compare with the sweep."""
    saved = next(
        row for row in load_results(DEV_SWEEP_PATH)
        if row["config_id"] == FROZEN_CONFIG["dev_config_id"]
    )
    i, restart = saved["image_index"], saved["restart"]
    hr = load_hr256(DATA_DIR / index_rows[i]["relative_path"])
    # The dev sweep used 3,000 steps; rerun with its step count to compare.
    metrics, _ = reconstruct(i, restart, int(index_rows[i]["label"]), lr_all[i], hr,
                             num_steps=saved["num_steps"])
    print(f"Dev image {i}, restart {restart}")
    for name in ("initial_loss", "final_loss", "psnr", "final_eps_rms"):
        print(f"  {name:>14}: saved {saved[name]:.4f}   rerun {metrics[name]:.4f}")
    print(f"  elapsed: {metrics['elapsed_seconds']:.1f}s (includes JIT compile)")


def build_run_plan(sample, labels):
    """All runs, image by image: oracle r0, r1, then wrong r0, r1."""
    plan = []
    for image_index in sample:
        image_index = int(image_index)
        true_label = int(labels[image_index])
        class_label = {"oracle": true_label, "wrong": wrong_label_for(image_index, true_label)}
        for condition in CONDITIONS:
            for restart in range(FROZEN_CONFIG["num_restarts"]):
                plan.append({
                    "image_index": image_index,
                    "condition": condition,
                    "restart": restart,
                    "true_label": true_label,
                    "class_label": class_label[condition],
                    **FROZEN_CONFIG,
                })
    return plan


def dry_run_summary(run_plan, sample, test_indices, results_path, manifest_path, arrays_dir, spec):
    """Validate the plan and report what a real run would do. Writes nothing."""
    assert all(run["condition"] in CONDITIONS for run in run_plan)
    assert np.all(np.isin(sample, test_indices)), "sample contains non-test images"
    for run in run_plan:
        if run["condition"] == "oracle":
            assert run["class_label"] == run["true_label"]
        else:
            assert run["class_label"] != run["true_label"], f"wrong label equals true label: {run}"
    expected = len(sample) * len(CONDITIONS) * FROZEN_CONFIG["num_restarts"]
    assert len(run_plan) == expected, (len(run_plan), expected)

    done = 0
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != spec:
            raise RuntimeError(f"Existing manifest does not match this plan: {manifest_path}")
        done = len(load_results(results_path)) if results_path.exists() else 0
    remaining = len(run_plan) - done
    free_gb = os.statvfs(DATA_DIR).f_bavail * os.statvfs(DATA_DIR).f_frsize / 1e9

    print("DRY RUN: nothing is run or written.")
    print(f"Images: {len(sample)} test images, {len(np.unique(sample))} distinct; conditions {CONDITIONS}; "
          f"restarts {FROZEN_CONFIG['num_restarts']}; steps {FROZEN_CONFIG['num_steps']:,}; lr {FROZEN_CONFIG['learning_rate']}")
    print(f"Planned runs: {len(run_plan)}; already saved: {done}; remaining: {remaining}")
    print(f"First runs: {[(r['image_index'], r['condition'], r['restart'], r['class_label']) for r in run_plan[:4]]}")
    print(f"Results:  {results_path}\nManifest: {manifest_path} ({'exists' if manifest_path.exists() else 'will be created'})")
    print(f"Arrays:   {arrays_dir}/ (~1.6 MB per run, ~{remaining * 1.6 / 1000:.1f} GB to write; {free_gb:.0f} GB free)")
    print(f"Estimated time at ~65 s/run: {remaining * 65 / 3600:.1f} h")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-images", type=int, default=TEST_SAMPLE_SIZE,
                        help=f"use the first N images of the fixed {TEST_SAMPLE_SIZE}-image test sample")
    parser.add_argument("--max-runs", type=int, default=32)
    parser.add_argument("--check-dev", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and validate the run plan, print a summary; run and write nothing")
    args = parser.parse_args()
    if not 1 <= args.num_images <= TEST_SAMPLE_SIZE:
        parser.error(f"--num-images must be between 1 and {TEST_SAMPLE_SIZE}")

    index_rows = load_index()
    labels = np.array([int(row["label"]) for row in index_rows])
    lr_all = np.load(DATA_DIR / "imageNet_x4.npy", mmap_mode="r")
    assert lr_all.shape == (10_000, 64, 64, 3)

    if args.check_dev:
        check_dev(index_rows, lr_all)
        return

    _, test_indices = load_split(labels)
    sample = get_test_sample(labels, test_indices)[: args.num_images]
    run_plan = build_run_plan(sample, labels)

    # The manifest pins everything that defines the experiment. Its hash names
    # the checkpoint files, so a different N (or any change) never mixes rows.
    spec = {
        "version": EXPERIMENT_VERSION,
        "model": "hf://pixel_B_sota",
        "operator": "A_sr4",
        "conditions": list(CONDITIONS),
        "sweep_seed": SWEEP_SEED,
        "test_sample_path": TEST_SAMPLE_PATH.name,
        "test_sample_seed": TEST_SAMPLE_SEED,
        "wrong_label_seed": WRONG_LABEL_SEED,
        "num_images": args.num_images,
        "run_plan": run_plan,
    }
    spec_text = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    plan_hash = hashlib.sha256(spec_text.encode("utf-8")).hexdigest()[:12]
    stem = DATA_DIR / f"phase1_{EXPERIMENT_VERSION}_n{args.num_images}_{plan_hash}"
    results_path = Path(f"{stem}.jsonl")
    manifest_path = Path(f"{stem}.manifest.json")
    arrays_dir = Path(f"{stem}_arrays")
    plan_by_key = {run_key(run): run for run in run_plan}
    assert len(plan_by_key) == len(run_plan)

    if args.dry_run:
        dry_run_summary(run_plan, sample, test_indices, results_path, manifest_path, arrays_dir, spec)
        return

    with Path(f"{stem}.lock").open("a+") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Checkpoint is locked by another process.") from error

        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != spec:
                raise RuntimeError("Manifest does not match the current run plan.")
        else:
            if results_path.exists():
                raise RuntimeError("Results exist without a manifest; refusing to mix runs.")
            atomic_write_bytes(manifest_path, (spec_text + "\n").encode("utf-8"))

        results = load_results(results_path) if results_path.exists() else []
        completed = {run_key(row) for row in results}
        if len(completed) != len(results) or not completed <= plan_by_key.keys():
            raise RuntimeError("Checkpoint has duplicate or unexpected runs.")
        remaining = [run for run in run_plan if run_key(run) not in completed]
        todo = remaining[: args.max_runs]

        print(f"Checkpoint: {results_path}")
        print(f"Completed: {len(completed)} / {len(run_plan)}; this call: {len(todo)}")

        for run in tqdm(todo, desc="CSGM test", unit="run", mininterval=10):
            i, condition, restart = run["image_index"], run["condition"], run["restart"]
            hr = load_hr256(DATA_DIR / index_rows[i]["relative_path"])
            metrics, arrays = reconstruct(i, restart, run["class_label"], lr_all[i], hr)

            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            array_path = arrays_dir / f"img{i:05d}_{condition}_r{restart}.npz"
            atomic_write_bytes(array_path, buffer.getvalue())

            result = {
                **run,
                **metrics,
                "bicubic_psnr": psnr(hr, bicubic_up(lr_all[i])),
                "array_file": array_path.name,
            }
            atomic_save_results(results_path, results + [result])
            results.append(result)
            tqdm.write(
                f"[{len(results)}/{len(run_plan)}] image {i} {condition:<6} r{restart} "
                f"(class {run['class_label']}): loss {metrics['final_loss']:.1f}, "
                f"PSNR {metrics['psnr']:.2f} dB (bicubic {result['bicubic_psnr']:.2f}), "
                f"{metrics['elapsed_seconds']:.0f}s"
            )

        print(f"Saved {len(results)} / {len(run_plan)} results to {results_path}")
        if len(results) == len(run_plan):
            print(f"ALL RUNS COMPLETE: {results_path}")


if __name__ == "__main__":
    main()
