"""Prepare the data: ImageNet-V2 MatchedFrequency, the measurements y = A_sr4(HR), the 80/20 split and the test sample.

Steps (each is skipped if its output exists; nothing is ever overwritten):
    1. download   imagenetv2-matched-frequency.tar.gz from the Hugging Face dataset vaishaal/ImageNetV2 and extract it
                  (10,000 images, 10 per class; the folder name is the class label)
    2. classes    imagenet_class_index.json (Keras class index; only used for readable class names)
    3. measure    imageNet_x4.npy: y = A_sr4(HR) for all 10,000 images (float32 [10000, 64, 64, 3]), and
                  imageNet_x4_index.csv (index, relative_path, label; images in sorted path order). HR = EXIF
                  orientation -> RGB -> bicubic short-side resize to 256 -> centre crop -> [-1, 1]
                  (iadrift.data.load_hr256); A_sr4 = antialiased bicubic 256 -> 64 (iadrift.operators.A_sr4)
    4. split      imageNetV2_split_80_20_seed2026.npz: per class 8 development + 2 test images (seed 2026)
    5. sample     sr4_test_sample_100_seed2028.npz: the 100 test images from 100 distinct classes (seed 2028)

--check recomputes everything that can be recomputed cheaply and compares it with the existing files (the index order,
the split, the test sample, the class-index checksum, and A_sr4 of --check-images images), writing nothing.

Usage:
    python scripts/data/prepare_data.py              # all steps (GPU recommended for step 3; ~10 min)
    python scripts/data/prepare_data.py --check      # verify existing files; writes nothing (CPU is fine)

Output directory: IADRIFT_DATA (default <repo>/data/ImageNetV2).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

import argparse
import csv
import hashlib
import io
import tarfile
import urllib.request

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import atomic_write_bytes

HF_REPO, HF_FILE = "vaishaal/ImageNetV2", "imagenetv2-matched-frequency.tar.gz"
IMAGE_DIR = "imagenetv2-matched-frequency-format-val"
CLASS_INDEX_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
CLASS_INDEX_MD5 = "c2c37ea517e94d9795004a39431a14cb"
BATCH = 32


def find_images(data_dir):
    return sorted(p for p in (data_dir / IMAGE_DIR).rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def step_download(data_dir):
    if len(find_images(data_dir)) == data.NUM_IMAGES:
        print(f"[download] {data.NUM_IMAGES} images present")
        return
    from huggingface_hub import hf_hub_download
    archive = Path(hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=HF_FILE, local_dir=data_dir))
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(data_dir, filter="data")
    assert len(find_images(data_dir)) == data.NUM_IMAGES, "unexpected number of images after extraction"
    print(f"[download] extracted {data.NUM_IMAGES} images into {data_dir / IMAGE_DIR}")


def step_classes(data_dir):
    path = data_dir / paths.CLASS_INDEX_JSON
    if not path.exists():
        with urllib.request.urlopen(CLASS_INDEX_URL) as response:
            atomic_write_bytes(path, response.read())
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    assert md5 == CLASS_INDEX_MD5, f"{path} checksum {md5} != {CLASS_INDEX_MD5}"
    print(f"[classes] {path.name} ok")


def index_rows_from_files(data_dir):
    images = find_images(data_dir)
    assert len(images) == data.NUM_IMAGES, f"found {len(images)} images"
    return [{"index": str(k), "relative_path": str(p.relative_to(data_dir)), "label": str(int(p.parent.name))}
            for k, p in enumerate(images)]


def measure(hr_paths):
    """A_sr4 of a list of HR images, in jitted batches of 32 (as in the original run)."""
    import jax
    import jax.numpy as jnp
    from iadrift.operators import A_sr4

    downsample = jax.jit(A_sr4)
    out = np.empty((len(hr_paths), 64, 64, 3), np.float32)
    for start in range(0, len(hr_paths), BATCH):
        batch = np.stack([data.load_hr256(p) for p in hr_paths[start:start + BATCH]])
        out[start:start + len(batch)] = np.asarray(jax.device_get(downsample(jnp.asarray(batch))), dtype=np.float32)
    return out


def step_measure(data_dir):
    npy, csv_path = data_dir / paths.MEASUREMENTS_NPY, data_dir / paths.INDEX_CSV
    if npy.exists() and csv_path.exists():
        print(f"[measure] {npy.name} and {csv_path.name} present")
        return
    from tqdm.auto import tqdm
    rows = index_rows_from_files(data_dir)
    partial = data_dir / "imageNet_x4.partial.npy"
    y = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(data.NUM_IMAGES, 64, 64, 3))
    for start in tqdm(range(0, data.NUM_IMAGES, BATCH * 10), desc="A_sr4"):
        chunk = rows[start:start + BATCH * 10]
        y[start:start + len(chunk)] = measure([data_dir / r["relative_path"] for r in chunk])
    y.flush()
    del y
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["index", "relative_path", "label"])
    for r in rows:
        writer.writerow([r["index"], r["relative_path"], r["label"]])
    atomic_write_bytes(csv_path, buffer.getvalue().encode())
    partial.replace(npy)
    print(f"[measure] wrote {npy} and {csv_path}")


def step_split(data_dir):
    path = data_dir / paths.SPLIT_NPZ
    labels = data.labels_of(data.load_index(data_dir))
    if not path.exists():
        train, test = data.draw_split(labels)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, train_indices=train, test_indices=test, seed=data.SPLIT_SEED)
        atomic_write_bytes(path, buffer.getvalue())
        print(f"[split] wrote {path}")
    data.load_split(labels, data_dir)  # checks the invariants
    print("[split] 8,000 development / 2,000 test images, 8 / 2 per class")


def step_sample(data_dir):
    path = data_dir / paths.TEST_SAMPLE_NPZ
    labels = data.labels_of(data.load_index(data_dir))
    _, test = data.load_split(labels, data_dir)
    if not path.exists():
        atomic_write_bytes(path, data.test_sample_bytes(data.draw_test_sample(labels, test), data_dir / paths.SPLIT_NPZ))
        print(f"[sample] wrote {path}")
    data.get_test_sample(labels, test, data_dir)  # checks it against a fresh seeded draw
    print(f"[sample] {data.TEST_SAMPLE_SIZE} test images from {data.TEST_SAMPLE_SIZE} classes")


def check(data_dir, n_images):
    ok = True
    saved_rows = data.load_index(data_dir)
    fresh_rows = index_rows_from_files(data_dir)
    same_index = [dict(r) for r in saved_rows] == fresh_rows
    print(f"[check] index (sorted file order, labels): {'identical' if same_index else 'DIFFERENT'}")
    ok &= same_index

    labels = data.labels_of(saved_rows)
    train, test = data.draw_split(labels)
    saved = np.load(data_dir / paths.SPLIT_NPZ)
    same_split = np.array_equal(saved["train_indices"], train) and np.array_equal(saved["test_indices"], test)
    print(f"[check] split: {'identical' if same_split else 'DIFFERENT'}")
    ok &= same_split

    sample = data.draw_test_sample(labels, test)
    same_sample = np.array_equal(np.load(data_dir / paths.TEST_SAMPLE_NPZ)["test_sample_indices"], sample)
    print(f"[check] test sample: {'identical' if same_sample else 'DIFFERENT'}")
    ok &= same_sample

    md5 = hashlib.md5((data_dir / paths.CLASS_INDEX_JSON).read_bytes()).hexdigest()
    print(f"[check] class index md5: {'ok' if md5 == CLASS_INDEX_MD5 else 'DIFFERENT'}")
    ok &= md5 == CLASS_INDEX_MD5

    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(data.NUM_IMAGES, size=n_images, replace=False))
    y_saved = data.load_measurements(data_dir)[idx]
    y_new = measure([data_dir / saved_rows[i]["relative_path"] for i in idx])
    diff = float(np.abs(y_new - y_saved).max())
    import jax
    # The saved measurements were computed on the GPU; on another backend the last float32 bits may differ.
    tol = 0.0 if jax.default_backend() == "gpu" else 1e-5
    print(f"[check] A_sr4 of {n_images} random images ({jax.default_backend()}): max |new - saved| = {diff:.2e} "
          f"(tolerance {tol:g})")
    ok &= diff <= tol
    print("CHECK PASSED" if ok else "CHECK FAILED")
    if not ok:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify the existing files; write nothing")
    parser.add_argument("--check-images", type=int, default=64, help="images to re-measure in --check")
    args = parser.parse_args()
    data_dir = paths.DATA_DIR
    if args.check:
        check(data_dir, args.check_images)
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    step_download(data_dir)
    step_classes(data_dir)
    step_measure(data_dir)
    step_split(data_dir)
    step_sample(data_dir)
    print(f"Data ready in {data_dir}")


if __name__ == "__main__":
    main()
