"""Export the development results on why adaptation fails to results/, where scripts/report/ reads them. CPU only.

Development images only, never test images: 8117 streetcar, 3939 bonnet, 403 crane, 347 spoonbill; correct label.
Reads the cached outputs of the development runs and computes all metrics on the CPU:

    results/dev_metrics.json    per image and method: PSNR, LPIPS, LR loss, band SNR, blockiness, patch-grid score;
                                in-range SR (R3) metrics; patch-grid scores of generator samples
    results/dev_traces.json     loss curves every 50 steps: theta-only adaptation (SR) and IADrift fitted to HR

Methods per image: bicubic, CSGM (restart with the lowest final LR loss) and IADrift (theta lr 1e-5, eps lr 1e-3,
1,000 steps), each raw and + BP; theta-only adaptation (theta lr 1e-6 and 1e-5); CSGM-HR (R1) and IADrift-HR
(theta lr 1e-5). In-range SR (R3) is scored against the generated target x*, not HR.

Reads from IADRIFT_DATA, written by these runners in scripts/dev/:
    csgm_dev_5000steps_lr0.02/      run_csgm_dev.py
    iadrift_dev_tuning_v1/          run_iadrift_dev.py
    theta_only_dev_v1/              run_theta_only_dev.py
    representation_error_dev_v1/    run_representation_dev.py   (R1, R3, summary.json)
    hr_fit_iadrift_dev_v1/          run_hr_fit_dev.py
    generator_samples_dev_v1.npz    run_generator_samples_dev.py

Usage:
    python scripts/analysis/export_dev_results.py            # write results/dev_*
    python scripts/analysis/export_dev_results.py --check    # write to a temporary directory, compare with results/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `iadrift` imports without install

from iadrift.runtime import setup_cpu

setup_cpu()

import argparse
import json
import tempfile

import jax.numpy as jnp
import numpy as np
import torch

from iadrift import data, paths
from iadrift.checkpoint import load_npz
from iadrift.metrics import band_snr_db, blockiness, grid_score, lpips_dist, psnr
from iadrift.operators import A_sr4, backproject, bicubic_up

DEV_IMAGES = list(data.DEV_IMAGES_4)
IAD_CFG = "lrt1e-05_lre0.001"
R3_CONDITIONS = ("noise labels known", "noise labels random")
FILES = ("dev_metrics.json", "dev_traces.json")

torch.set_num_threads(4)  # LPIPS values depend slightly on the thread count; results/ was written with 4


def evaluate(x, ref, y):
    # LR loss summed over the [64, 64, 3] array
    lr_loss = float(jnp.sum((A_sr4(jnp.asarray(x, jnp.float32)[None])[0] - jnp.asarray(y, jnp.float32)) ** 2))
    return {"psnr": psnr(ref, x), "lpips": lpips_dist(x, ref), "lr_loss": lr_loss, "band_snr_db": band_snr_db(x, ref),
            "blockiness": blockiness(x), "grid_score": grid_score(x)}


def best_of(file_paths):
    """The run with the lowest final loss (ties: lowest restart): (x_hat, restart)."""
    runs = []
    for r, p in enumerate(file_paths):
        metrics, arrays = load_npz(p)
        runs.append((metrics["final_loss"], r, arrays["x_hat"]))
    _, r, x = min(runs, key=lambda t: (t[0], t[1]))
    return x, r


def export(out_dir):
    D = paths.DATA_DIR
    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    names = data.class_names()
    _, test_indices = data.load_split(labels)
    assert not np.any(np.isin(DEV_IMAGES, test_indices)), "development images only"
    lr_all = data.load_measurements()
    gen = np.load(D / "generator_samples_dev_v1.npz")
    r3_summary = json.loads((D / "representation_error_dev_v1" / "summary.json").read_text())["R3"]

    metrics, traces = {"images": {}, "r3": {}, "samples_grid_score": {}}, {"theta_only": {}, "hr_fit": {}}
    for i in DEV_IMAGES:
        label = int(labels[i])
        hr, y = data.hr_image(index_rows, i), np.asarray(lr_all[i], np.float32)
        out = {"bicubic": bicubic_up(y)}
        out["CSGM"], csgm_r = best_of([D / "csgm_dev_5000steps_lr0.02" / f"img{i:05d}_oracle_r{r}.npz" for r in (0, 1)])
        with np.load(D / "iadrift_dev_tuning_v1" / f"img{i:05d}_{IAD_CFG}.npz") as d:
            out["IADrift"] = d["IADrift"]
        for b in ("bicubic", "CSGM", "IADrift"):
            out[f"{b} + BP"] = backproject(out[b], y)
        for lt in ("1e-06", "1e-05"):
            with np.load(D / "theta_only_dev_v1" / f"img{i:05d}_lrt{lt}_steps2000.npz") as d:
                out[f"theta-only, lr {lt}"] = d["x"]
                traces["theta_only"][f"{i}|{lt}"] = json.loads(str(d["result_json"]))["trace"]
        out["CSGM-HR"], _ = best_of([D / "representation_error_dev_v1" / f"R1_img{i:05d}_r{r}.npz" for r in (0, 1)])
        for lt in ("1e-05",):
            with np.load(D / "hr_fit_iadrift_dev_v1" / f"img{i:05d}_lrt{lt}_lre0.001_steps2000.npz") as d:
                out[f"IADrift-HR, lr {lt}"] = d["x"]
                traces["hr_fit"][f"{i}|{lt}"] = json.loads(str(d["result_json"]))["trace"]
        metrics["images"][str(i)] = {"label": label, "class": names[label], "csgm_restart": csgm_r,
                                     "HR": {"blockiness": blockiness(hr), "grid_score": grid_score(hr)},
                                     **{m: evaluate(x, hr, y) for m, x in out.items()}}

        # In-range SR (R3): targets from the generator itself, scored against x*.
        x_star = gen[f"xstar_img{i}"]
        y_star = np.asarray(A_sr4(jnp.asarray(x_star)[None])[0])
        r3 = {"bicubic": bicubic_up(y_star)}
        r3["bicubic + BP"] = backproject(r3["bicubic"], y_star)
        for cond in R3_CONDITIONS:
            slug = cond.replace(" ", "_")
            r3[f"CSGM, {cond}"], _ = best_of([D / "representation_error_dev_v1" / f"R3_img{i:05d}_{slug}_r{r}.npz"
                                              for r in (0, 1)])
        metrics["r3"][str(i)] = {m: evaluate(x, x_star, y_star) for m, x in r3.items()}
        for cond in R3_CONDITIONS:
            metrics["r3"][str(i)][f"CSGM, {cond}"]["eps_rel_error"] = r3_summary[str(i)][f"eps_err, {cond}"]
        metrics["samples_grid_score"][str(i)] = [grid_score(gen[f"sample_img{i}_draw{d}_cfg1"]) for d in (0, 1)]
        print(f"image {i} done: CSGM {metrics['images'][str(i)]['CSGM']['psnr']:.2f} dB, "
              f"IADrift-HR 1e-5 {metrics['images'][str(i)]['IADrift-HR, lr 1e-05']['psnr']:.2f} dB", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dev_metrics.json").write_text(json.dumps(metrics, indent=1))
    (out_dir / "dev_traces.json").write_text(json.dumps(traces))
    for f in FILES:
        print(f"{f:20s} {(out_dir / f).stat().st_size / 1e6:6.2f} MB")


def max_abs_diff(a, b, where=""):
    """(max |a - b| over all numbers, list of structural differences) for nested JSON values."""
    if isinstance(a, dict) and isinstance(b, dict):
        worst, issues = 0.0, ([f"{where}: keys differ"] if a.keys() != b.keys() else [])
        for k in a.keys() & b.keys():
            w, s = max_abs_diff(a[k], b[k], f"{where}/{k}")
            worst, issues = max(worst, w), issues + s
        return worst, issues
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return 0.0, [f"{where}: lengths differ"]
        results = [max_abs_diff(x, z, f"{where}[{n}]") for n, (x, z) in enumerate(zip(a, b))]
        return max([0.0] + [w for w, _ in results]), [s for _, ss in results for s in ss]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b), []
    return 0.0, ([] if a == b else [f"{where}: {a!r} != {b!r}"])


def check(reference_dir):
    with tempfile.TemporaryDirectory(prefix="export_dev_check_") as tmp:
        tmp = Path(tmp)
        export(tmp)
        ok = True
        for f in ("dev_metrics.json", "dev_traces.json"):
            new, ref = (tmp / f).read_text(), (reference_dir / f).read_text()
            worst, issues = max_abs_diff(json.loads(new), json.loads(ref))
            print(f"{f}: byte-identical {new == ref}; max |difference| {worst:.3g}; "
                  f"structural differences {issues[:5]}")
            ok &= new == ref
    print("CHECK PASSED" if ok else "CHECK FAILED")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="export to a temporary directory and compare with results/")
    parser.add_argument("--out", type=Path, default=paths.RESULTS_DIR, help="output dir; the reference dir with --check")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if check(args.out) else 1)
    export(args.out)


if __name__ == "__main__":
    main()
