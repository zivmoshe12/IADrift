"""ImageNet-V2 data: HR loading, the image index, the 80/20 split, the fixed test sample, wrong labels.

All files are created once by ``scripts/data/prepare_data.py`` and only read here (the readers check the invariants
the experiments rely on). No JAX import, so the analysis scripts can use this module on a CPU.
"""

import csv
import io
import json

import numpy as np
from PIL import Image, ImageOps

from . import paths

NUM_IMAGES = 10_000
NUM_CLASSES = 1_000

SPLIT_SEED = 2026          # 80/20 split, 8 development and 2 test images per class
SWEEP_SEED = 2026          # CSGM initialization key(SWEEP_SEED, image, restart); see generator.initialize
TEST_SAMPLE_SEED = 2028    # the 100-image test sample
TEST_SAMPLE_SIZE = 100
WRONG_LABEL_SEED = 2029    # one random wrong class per test image (secondary analysis)

# Development images used by the development experiments (never test images).
DEV_IMAGES_4 = (8117, 3939, 403, 347)                  # streetcar, bonnet, crane, spoonbill (Checks 1-12)
DEV_IMAGES_6 = DEV_IMAGES_4 + (9780, 8405)             # + two fresh images (Checks 10b-14)


def load_hr256(path):
    """Load an image as a centred 256x256 RGB array in [-1, 1].

    EXIF orientation -> RGB -> bicubic resize of the short side to 256 -> centre crop.
    """
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


def load_index(data_dir=None):
    """The 10,000 rows (index, relative_path, label), in index order."""
    data_dir = data_dir or paths.DATA_DIR
    with (data_dir / paths.INDEX_CSV).open(newline="") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: int(row["index"]))
    assert [int(row["index"]) for row in rows] == list(range(NUM_IMAGES))
    return rows


def labels_of(index_rows):
    return np.array([int(row["label"]) for row in index_rows])


def load_measurements(data_dir=None):
    """y = A_sr4(HR) for every image, memory-mapped float32 [10000, 64, 64, 3]."""
    data_dir = data_dir or paths.DATA_DIR
    lr_all = np.load(data_dir / paths.MEASUREMENTS_NPY, mmap_mode="r")
    assert lr_all.shape == (NUM_IMAGES, 64, 64, 3)
    return lr_all


def hr_image(index_rows, image_index, data_dir=None):
    data_dir = data_dir or paths.DATA_DIR
    return load_hr256(data_dir / index_rows[int(image_index)]["relative_path"])


def load_split(labels, data_dir=None):
    """Read (never rewrite) the 80/20 split and check its invariants. Returns (development, test) indices."""
    data_dir = data_dir or paths.DATA_DIR
    split = np.load(data_dir / paths.SPLIT_NPZ)
    train, test = split["train_indices"], split["test_indices"]
    assert int(split["seed"]) == SPLIT_SEED
    assert len(train) == 8_000 and len(test) == 2_000
    assert np.intersect1d(train, test).size == 0
    assert np.all(np.bincount(labels[test], minlength=NUM_CLASSES) == 2)
    return train, test


def draw_split(labels):
    """The stratified 80/20 split (8 development + 2 test images per class), seed SPLIT_SEED."""
    class_ids, class_counts = np.unique(labels, return_counts=True)
    assert len(class_ids) == NUM_CLASSES and np.all(class_counts == 10), "Expected 10 images per class."
    rng = np.random.default_rng(SPLIT_SEED)
    train, test = [], []
    for class_id in class_ids:
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        test.extend(class_indices[:2])
        train.extend(class_indices[2:])
    train, test = np.asarray(train, dtype=np.int32), np.asarray(test, dtype=np.int32)
    rng.shuffle(train)  # shuffle the order within each split, without changing membership
    rng.shuffle(test)
    assert np.intersect1d(train, test).size == 0
    assert np.unique(np.concatenate([train, test])).size == len(labels)
    return train, test


def draw_test_sample(labels, test_indices):
    """One test image from each of TEST_SAMPLE_SIZE distinct classes (seed TEST_SAMPLE_SEED)."""
    rng = np.random.default_rng(TEST_SAMPLE_SEED)
    classes = rng.choice(np.unique(labels[test_indices]), size=TEST_SAMPLE_SIZE, replace=False)
    sample = np.asarray([int(rng.choice(test_indices[labels[test_indices] == c])) for c in classes], dtype=np.int32)
    assert len(np.unique(labels[sample])) == TEST_SAMPLE_SIZE
    assert np.all(np.isin(sample, test_indices))
    return sample


def get_test_sample(labels, test_indices, data_dir=None):
    """The fixed 100-image test sample (sample order = analysis order). Checked against a fresh seeded draw."""
    data_dir = data_dir or paths.DATA_DIR
    sample = draw_test_sample(labels, test_indices)
    saved = np.load(data_dir / paths.TEST_SAMPLE_NPZ)["test_sample_indices"]
    if not np.array_equal(saved, sample):
        raise RuntimeError(f"{paths.TEST_SAMPLE_NPZ} differs from the seeded draw.")
    return sample


def test_sample_bytes(sample, split_path):
    """The .npz written by prepare_data (same fields as the original run)."""
    buffer = io.BytesIO()
    np.savez(buffer, test_sample_indices=sample, seed=TEST_SAMPLE_SEED, split_path=str(split_path))
    return buffer.getvalue()


def wrong_label_for(image_index, true_label):
    """A uniformly random class != true_label, fixed per image (seeded by WRONG_LABEL_SEED and the image index)."""
    rng = np.random.default_rng([WRONG_LABEL_SEED, int(image_index)])
    offset = int(rng.integers(1, NUM_CLASSES))  # 1..999, never 0
    return (int(true_label) + offset) % NUM_CLASSES


def class_names(data_dir=None):
    """{label: readable class name} from the Keras ImageNet class index."""
    data_dir = data_dir or paths.DATA_DIR
    index = json.loads((data_dir / paths.CLASS_INDEX_JSON).read_text())
    return {int(k): v[1].replace("_", " ") for k, v in index.items()}
