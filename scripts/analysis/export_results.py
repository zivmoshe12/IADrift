"""Export the compact CSGM and IADrift test results to results/, where scripts/report/ reads them. CPU only.

The full run outputs (~1 GB of arrays per run family) stay outside the repository. This script reads the run caches
and the evaluate_test.py caches from IADRIFT_DATA and writes:

    results/test_sample.json        the 100 test images: index, true and wrong class (label and name), kept CSGM restart
    results/test_metrics.json       per-image metrics (PSNR, LPIPS, blockiness, LR loss, band SNR) of every method,
                                    correct and wrong label, from csgm_test_postproc_v1 and iadrift_test_postproc_v1
    results/primary_iadrift.json    the 4 pre-registered primary tests (copy of iadrift_test_postproc_v1/primary.json)
    results/iadrift_traces.npz      IADrift LR loss over steps for every test image and label condition
    results/examples_test.npz       the first 4 test images in sample order: LR, HR and every method's output, float16,
                                    for figures only

Usage:
    python scripts/analysis/export_results.py --check    # write to a temporary directory, compare with results/
    python scripts/analysis/export_results.py            # rebuild results/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `iadrift` imports without install

from iadrift.runtime import setup_cpu

setup_cpu()  # JAX_PLATFORMS=cpu

import argparse
import json
import shutil
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import load_results

CSGM_STEM = "phase1_sr4_csgm_label_test_v2_n100_9e24c78a47ab"
IAD_STEM = "phase2_sr4_iadrift_test_v1_n100_00ca59364e18"
COND = {"oracle": "correct label", "wrong": "wrong label"}
NUM_EXAMPLES = 4
FILES = ("test_sample.json", "test_metrics.json", "primary_iadrift.json", "iadrift_traces.npz", "examples_test.npz")


def export(out):
    D = paths.DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    name = data.class_names()
    sample = [int(i) for i in np.load(D / paths.TEST_SAMPLE_NPZ)["test_sample_indices"]]

    csgm = {(r["image_index"], r["condition"], r["restart"]): r for r in load_results(D / f"{CSGM_STEM}.jsonl")}
    iad = {(r["image_index"], r["condition"]): r for r in load_results(D / f"{IAD_STEM}.jsonl")}
    assert len(csgm) == 400 and len(iad) == 200
    kept = {(i, c): min((0, 1), key=lambda r: (csgm[(i, c, r)]["final_loss"], r)) for i in sample for c in COND}
    for i in sample:  # IADrift must have started from the kept CSGM restart
        for c in COND:
            assert iad[(i, c)]["csgm_restart"] == kept[(i, c)]

    info = [{"image_index": i, "true_label": csgm[(i, "oracle", 0)]["true_label"],
             "true_class": name[csgm[(i, "oracle", 0)]["true_label"]],
             "wrong_label": csgm[(i, "wrong", 0)]["class_label"],
             "wrong_class": name[csgm[(i, "wrong", 0)]["class_label"]],
             "csgm_kept_restart": {COND[c]: kept[(i, c)] for c in COND}} for i in sample]
    (out / "test_sample.json").write_text(json.dumps(info, indent=1))

    csgm_post = json.loads((D / "csgm_test_postproc_v1" / "metrics.json").read_text())
    iad_post = json.loads((D / "iadrift_test_postproc_v1" / "metrics.json").read_text())
    metrics = {str(i): {"hr_blockiness": csgm_post[str(i)]["hr_blockiness"],
                        **{m: v for m, v in csgm_post[str(i)].items() if m != "hr_blockiness"},
                        **iad_post[str(i)]} for i in sample}
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "primary_iadrift.json").write_text((D / "iadrift_test_postproc_v1" / "primary.json").read_text())

    traces, steps = {}, None
    for c in COND:
        t = []
        for i in sample:
            with np.load(D / f"{IAD_STEM}_arrays" / iad[(i, c)]["array_file"]) as d:
                steps = d["trace_steps"]
                t.append(d["trace_loss"])
        traces[c] = np.stack(t).astype(np.float32)
    start = {c: np.array([iad[(i, c)]["start_lr_loss"] for i in sample], np.float32) for c in COND}
    np.savez_compressed(out / "iadrift_traces.npz", steps=steps, image_index=np.array(sample),
                        loss_oracle=traces["oracle"], loss_wrong=traces["wrong"],
                        start_loss_oracle=start["oracle"], start_loss_wrong=start["wrong"])

    index_rows = data.load_index()
    lr_all = data.load_measurements()
    up = jax.jit(lambda y: jax.image.resize(y, (256, 256, 3), method="cubic", antialias=False))  # as in evaluate_test
    ex = {}
    for k, i in enumerate(sample[:NUM_EXAMPLES]):
        x = {"lr": np.asarray(lr_all[i], np.float32), "hr": data.hr_image(index_rows, i)}
        x["bicubic"] = np.asarray(up(jnp.asarray(x["lr"])))
        with np.load(D / "csgm_test_postproc_v1" / f"img{i:05d}.npz") as d:
            x["bicubic + BP"] = d["bicubic_BP"]
            for c in COND:
                x[f"CSGM, {COND[c]} + BP"] = d[f"CSGM_{COND[c].replace(' ', '_')}_BP"]
        with np.load(D / "iadrift_test_postproc_v1" / f"img{i:05d}.npz") as d:
            for c in COND:
                for s in (100, 1000):
                    x[f"IADrift @{s}, {COND[c]} + BP"] = d[f"bp_{c}_step{s}"]
        for c in COND:
            with np.load(D / f"{CSGM_STEM}_arrays" / f"img{i:05d}_{c}_r{kept[(i, c)]}.npz") as d:
                x[f"CSGM, {COND[c]}"] = d["x_hat"]
            with np.load(D / f"{IAD_STEM}_arrays" / iad[(i, c)]["array_file"]) as d:
                for s in (100, 1000):
                    x[f"IADrift @{s}, {COND[c]}"] = d[f"x_step{s}"]
        for m, a in x.items():
            ex[f"{k}|{m}"] = np.asarray(a, np.float16)
    np.savez_compressed(out / "examples_test.npz", image_index=np.array(sample[:NUM_EXAMPLES]), **ex)
    for f in FILES:
        print(f"{f:28s} {(out / f).stat().st_size / 1e6:6.2f} MB")


# ---------------------------------------------------------------- --check (shared with export_ensemble_results.py)
def _compare_json(new, old, path, s, rtol):
    if isinstance(old, dict) or isinstance(old, list):
        keys = list(old) if isinstance(old, dict) else range(len(old))
        if type(new) is not type(old) or (list(new) if isinstance(new, dict) else len(new)) != \
                (list(old) if isinstance(old, dict) else len(old)):
            s["mismatch"].append(f"{path}: structure / key order")
            return
        for k in keys:
            _compare_json(new[k], old[k], f"{path}/{k}", s, rtol)
    elif isinstance(old, float) or (isinstance(old, int) and not isinstance(old, bool) and isinstance(new, float)):
        s["n"] += 1
        rel = abs(new - old) / max(abs(old), 1e-300)
        s["exact"] += new == old
        s["max_rel"] = max(s["max_rel"], rel)
        if rel > rtol:
            s["mismatch"].append(f"{path}: {new!r} vs {old!r}")
    else:
        s["n"] += 1
        s["exact"] += new == old
        if new != old:
            s["mismatch"].append(f"{path}: {new!r} vs {old!r}")


def compare_dirs(new_dir, old_dir, files, rtol=1e-12):
    """JSON: same keys in the same order, values equal within rtol. .npz: same names, dtypes and values."""
    ok = True
    for f in files:
        new, old = new_dir / f, old_dir / f
        if not old.exists():
            print(f"  {f}: no existing file in {old_dir}  <-- MISSING")
            ok = False
            continue
        if f.endswith(".json"):
            s = {"n": 0, "exact": 0, "max_rel": 0.0, "mismatch": []}
            _compare_json(json.loads(new.read_text()), json.loads(old.read_text()), "", s, rtol)
            same_bytes = new.read_bytes() == old.read_bytes()
            print(f"  {f}: {s['n']} values, {s['exact']} identical, max relative difference {s['max_rel']:.2e}; "
                  f"byte-identical file: {same_bytes}" + ("" if not s["mismatch"] else "  <-- DIFFERS"))
            for m in s["mismatch"][:5]:
                print(f"      {m}")
            ok &= not s["mismatch"]
        else:
            with np.load(new) as a, np.load(old) as b:
                names_ok = sorted(a.files) == sorted(b.files)
                diff = [k for k in b.files if k not in a.files or a[k].dtype != b[k].dtype
                        or not np.array_equal(a[k], b[k])]
                max_abs = max([float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max())
                               for k in diff if k in a.files and a[k].shape == b[k].shape] or [0.0])
            print(f"  {f}: {len(b.files)} arrays, {len(b.files) - len(diff)} identical (name, dtype, values); "
                  f"same names: {names_ok}; max abs difference {max_abs:.2e}" + ("" if not diff else "  <-- DIFFERS"))
            ok &= names_ok and not diff
    return ok


def run(export_fn, files, description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=paths.RESULTS_DIR, help="output directory (default: results/)")
    parser.add_argument("--check", action="store_true",
                        help="write to a temporary directory and compare with the files in --out; writes nothing else")
    args = parser.parse_args()
    if not args.check:
        export_fn(args.out)
        return
    tmp = Path(tempfile.mkdtemp(prefix="export_check_"))
    try:
        export_fn(tmp)
        print(f"\n[check] {tmp} vs {args.out}")
        ok = compare_dirs(tmp, args.out, files)
    finally:
        shutil.rmtree(tmp)
    print("CHECK PASSED" if ok else "CHECK FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run(export, FILES, __doc__)
