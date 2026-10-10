"""Export the posterior-mean ensemble test results (docs/PROTOCOL.md, Section 11, Addendum A) into results/. CPU only.

    results/ensemble_test_metrics.json   per test image (100): metrics of CSGM + BP and of the ensembles E1, E2, Ens-U, Ens-W
                                         (+ BP); the 5 members' LR losses and MSEs, the ensemble weights, and the MSE of
                                         each ensemble before BP (for the MMSE check)
    results/ensemble_primary.json        the 2 primary tests (Holm), as computed by evaluate_test.py --part ensemble
    results/ensemble_examples.npz        first 4 test images (fixed sample order): the 5 members (shifted back to the
                                         original frame), Ens-U + BP and Ens-W + BP, float16

Sources (IADRIFT_DATA): csgm_test_ensemble_postproc_v1/ (scripts/analysis/evaluate_test.py --part ensemble),
csgm_test_ensemble_v1/ (shifted CSGM runs, scripts/test/run_ensemble_test.py),
phase1_sr4_csgm_label_test_v2_n100_9e24c78a47ab_arrays/ (restarts r0, r1, scripts/test/run_csgm_test.py).

Ported from scripts/export_ensemble_results.py in our working repository.

Usage:
    python scripts/analysis/export_ensemble_results.py --check   # write to a temporary directory, compare with results/
    python scripts/analysis/export_ensemble_results.py           # rebuild the three files in results/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install
sys.path.insert(0, str(Path(__file__).resolve().parent))      # export_results (shared --check code)

import json

import numpy as np

from export_results import run  # also calls iadrift.runtime.setup_cpu()
from iadrift import paths
from iadrift.ensemble import SHIFTS, unshift_x

POST = "csgm_test_ensemble_postproc_v1"
RUNS = "csgm_test_ensemble_v1"
CSGM_ARRAYS = "phase1_sr4_csgm_label_test_v2_n100_9e24c78a47ab_arrays"
NUM_EXAMPLES = 4
FILES = ("ensemble_test_metrics.json", "ensemble_primary.json", "ensemble_examples.npz")


def export(out):
    D = paths.DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    sample = [int(i) for i in np.load(D / paths.TEST_SAMPLE_NPZ)["test_sample_indices"]]
    metrics = json.loads((D / POST / "metrics.json").read_text())
    assert sorted(map(int, metrics)) == sorted(sample), "the ensemble analysis must cover all 100 test images"
    (out / "ensemble_test_metrics.json").write_text(json.dumps({str(i): metrics[str(i)] for i in sample}, indent=1))
    (out / "ensemble_primary.json").write_text((D / POST / "primary.json").read_text())

    ex = {}
    for k, i in enumerate(sample[:NUM_EXAMPLES]):
        for r in (0, 1):
            with np.load(D / CSGM_ARRAYS / f"img{i:05d}_oracle_r{r}.npz") as d:
                ex[f"{k}|r{r}"] = d["x_hat"]
        for sy, sx in SHIFTS:
            with np.load(D / RUNS / f"img{i:05d}_shift{sy}{sx}_r0.npz") as d:
                ex[f"{k}|shift{sy}{sx}"] = unshift_x(d["x_hat"], sy, sx)
        with np.load(D / POST / f"img{i:05d}.npz") as d:
            ex[f"{k}|Ens-U + BP"], ex[f"{k}|Ens-W + BP"] = d["Ens-U_BP"], d["Ens-W_BP"]
    np.savez_compressed(out / "ensemble_examples.npz", image_index=np.array(sample[:NUM_EXAMPLES]),
                        **{key: np.asarray(x, np.float16) for key, x in ex.items()})
    for f in FILES:
        print(f"{f:28s} {(out / f).stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    run(export, FILES, __doc__)
