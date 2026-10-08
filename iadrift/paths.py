"""File locations. Every location can be set with an environment variable; no JAX import here.

    IADRIFT_DATA       ImageNet-V2 MatchedFrequency, the derived measurement files and every experiment cache
                       (default: <repo>/data/ImageNetV2)
    DRIFTING_DIR       a clone of https://github.com/lambertae/drifting (default: <repo>/third_party/drifting)
    IADRIFT_HF_CACHE   where the pretrained pixel_B_sota weights are downloaded (default: <repo>/hf_cache)

Example (the layout the experiments were run with):

    export IADRIFT_DATA=~/work/datasets/ImageNetV2 DRIFTING_DIR=~/work/drifting IADRIFT_HF_CACHE=~/work/hf_cache
"""

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent


def _env_path(name, default):
    return Path(os.environ.get(name, default)).expanduser()


DATA_DIR = _env_path("IADRIFT_DATA", REPO_DIR / "data" / "ImageNetV2")
DRIFTING_DIR = _env_path("DRIFTING_DIR", REPO_DIR / "third_party" / "drifting")
HF_CACHE_DIR = _env_path("IADRIFT_HF_CACHE", REPO_DIR / "hf_cache")
RESULTS_DIR = REPO_DIR / "results"
FIGURES_DIR = REPO_DIR / "figures"

# Files created by scripts/data/prepare_data.py (names kept from the original runs, so caches stay valid).
INDEX_CSV = "imageNet_x4_index.csv"            # index, relative_path, label for the 10,000 images
MEASUREMENTS_NPY = "imageNet_x4.npy"           # y = A_sr4(HR) for every image, float32 [10000, 64, 64, 3]
SPLIT_NPZ = "imageNetV2_split_80_20_seed2026.npz"
TEST_SAMPLE_NPZ = "sr4_test_sample_100_seed2028.npz"
CLASS_INDEX_JSON = "imagenet_class_index.json"
