"""Test-set evaluation on the 100 test images, CPU only.

Reads the saved GPU runs and computes the per-image metrics and paired tests behind the reported test numbers.

  --part csgm
      Restart selection: for each image and label condition, the CSGM restart with the lowest final LR loss (never HR).
      Methods: bicubic, CSGM with the correct label, CSGM with a wrong label; each raw and + BP.
  --part iadrift   (pre-registered)
      IADrift at steps 100 and 1,000, correct and wrong label, raw and + BP. Primary analysis (protocol Section 6):
      IADrift @1000 + BP and @100 + BP vs CSGM + BP, correct label, on PSNR and LPIPS; 4 tests, Holm-corrected.
  --part ensemble  (protocol Section 11, post-hoc)
      Members: the saved CSGM test restarts r0, r1 and the 3 shifted CSGM runs, shifted back. Ensembles: E1, E2,
      Ens-U (uniform) and Ens-W, w_k ~ exp(-(L_k - L_min) / std L) with L_k = ||y - A x_k||^2; each + BP, against
      CSGM + BP. Primary analysis: Ens-W + BP vs CSGM + BP on PSNR and LPIPS, Holm over 2. Also stores, for the MMSE
      check, each member's MSE to HR and each ensemble's MSE and LR loss before BP.

Back-projection: BP(x) = x + A^T (A A^T)^{-1} (y - A x), solved by CG (tol 1e-6, at most 500 iterations).
Metrics per method and image: PSNR (data range 2), clipped PSNR, LPIPS (AlexNet, inputs clipped to [-1, 1]),
blockiness (16-pixel tile seams, ~1 means none), LR loss ||y - A x||^2, and band SNR in 4 radial bands.
Paired tests on the per-image differences d_i: mean with a normal 95% interval, mean +- 1.96 s / sqrt(n) where s is the
sample standard deviation (ddof 1); Wilcoxon signed-rank p; number of images where A is better. Secondary comparisons
are exploratory and uncorrected; they are printed but not stored.

Usage:
    python scripts/analysis/evaluate_test.py --check                       # recompute all parts into a temporary
                                                                           # directory and compare with the caches
    python scripts/analysis/evaluate_test.py --check --part csgm --num-images 5
    python scripts/analysis/evaluate_test.py --part all --out DIR          # write fresh caches under DIR

Outputs, one directory per part under IADRIFT_DATA by default. An existing cache is not overwritten without --overwrite.
    csgm_test_postproc_v1/           metrics.json ({image: {"hr_blockiness", method: metrics}}),
                                     img#####.npz (metrics_json, bicubic_BP, CSGM_correct_label_BP, CSGM_wrong_label_BP)
    iadrift_test_postproc_v1/        metrics.json, primary.json (the 4 primary tests),
                                     img#####.npz (metrics_json, bp_{oracle,wrong}_step{100,1000})
    csgm_test_ensemble_postproc_v1/  metrics.json, with an "info" entry per image: member LR losses, kept restart,
                                     weights, n_eff, mse_raw, lr_loss_raw, member_mse;
                                     primary.json (2 tests, written only for all 100 images);
                                     img#####.npz (metrics_json, E1_BP, E2_BP, Ens-U_BP, Ens-W_BP, CSGM_BP)
The iadrift and ensemble parts take the CSGM part's results from DIR/csgm_test_postproc_v1 if it exists, for example
when computed in the same call, and from IADRIFT_DATA otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `iadrift` imports without install

from iadrift.runtime import setup_cpu

setup_cpu()  # JAX_PLATFORMS=cpu

import argparse
import hashlib
import json
import math
import shutil
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from iadrift import data, paths
from iadrift.checkpoint import atomic_write_bytes, load_npz, load_results, save_npz
from iadrift.ensemble import MEMBERS, SHIFTS, unshift_x
from iadrift.metrics import BANDS, band_snr_db, blockiness, lpips_dist, psnr, psnr_of_mean_mse
from iadrift.operators import A_sr4, backproject, lr_loss
from iadrift.stats import holm

CSGM_STEM = "phase1_sr4_csgm_label_test_v2_n100_9e24c78a47ab"
IAD_STEM = "phase2_sr4_iadrift_test_v1_n100_00ca59364e18"
ENS_RUN_DIR = "csgm_test_ensemble_v1"
ENS_HASH = "ca2d0ec5e6ff"
PART_DIRS = {"csgm": "csgm_test_postproc_v1", "iadrift": "iadrift_test_postproc_v1",
             "ensemble": "csgm_test_ensemble_postproc_v1"}
BAND_NAMES = [name for name, *_ in BANDS]
CONDITIONS = ("oracle", "wrong")
COND_NAME = {"oracle": "correct label", "wrong": "wrong label"}
CKPTS = (100, 1000)

CSGM_METHODS = ["bicubic", "bicubic + BP", "CSGM, correct label", "CSGM, correct label + BP",
                "CSGM, wrong label", "CSGM, wrong label + BP"]
IAD_METHODS = [f"IADrift @{s}, {COND_NAME[c]}{bp}" for c in CONDITIONS for s in CKPTS for bp in ("", " + BP")]
ENSEMBLES = {"E1: mean of 2 restarts": ["r0", "r1"], "E2: mean of r0 + 3 shifts": ["r0"] + list(MEMBERS[2:]),
             "Ens-U: mean of 5 CSGM runs": list(MEMBERS), "Ens-W: weighted 5 CSGM runs": list(MEMBERS)}
ENS_METHODS = [n + " + BP" for n in ENSEMBLES]

# Same resize as iadrift.operators.bicubic_up, but jitted; the saved caches were computed this way.
_upsample = jax.jit(lambda y: jax.image.resize(y, (256, 256, 3), method="cubic", antialias=False))


def bicubic(y):
    return np.asarray(_upsample(jnp.asarray(y, jnp.float32)))


def mse(hr, x):
    return float(np.mean((np.asarray(hr, np.float64) - np.asarray(x, np.float64)) ** 2))


def evaluate(x, hr, y):
    """Same metrics as iadrift.metrics.evaluate, in the key order of the saved caches."""
    return {"psnr": psnr(hr, x), "psnr_clipped": psnr(hr, np.clip(x, -1, 1)), "lpips": lpips_dist(x, hr),
            "blockiness": blockiness(x), "lr_loss": lr_loss(x, y), "band_snr_db": band_snr_db(x, hr)}


def paired_stats(d, key):
    """Like iadrift.stats.paired_stats, except that p = 1 when all d are 0."""
    lower_better = key in ("lpips", "blockiness")
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    return {"mean": d.mean(), "lo": d.mean() - half, "hi": d.mean() + half,
            "p": stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0,
            "better": int(np.sum(d < 0 if lower_better else d > 0)), "n": len(d)}


def table(sources, images):
    """v(method, key, band=None) -> per-image values; sources: [(predicate on the method name, {image: metrics})]."""
    def v(method, key, band=None):
        src = next(s for match, s in sources if match(method))
        return np.array([src[i][method]["band_snr_db"][band] if band else src[i][method][key] for i in images])
    return v


def primary(tests, v):
    """[(A, B, key)] -> the stored primary summary (paired stats of A - B, Holm over the tests)."""
    res = [paired_stats(v(a, key) - v(b, key), key) for a, b, key in tests]
    adj = holm(np.array([r["p"] for r in res]))
    return [{"comparison": f"{a} − {b}", "metric": key, **{k: float(x) for k, x in r.items()}, "holm_p": float(pa)}
            for (a, b, key), r, pa in zip(tests, res, adj)]


def print_primary(title, summary):
    print(f"\n{title}")
    for k, r in enumerate(summary, start=1):
        f = "{:+.3f}" if r["metric"] == "lpips" else "{:+.2f}"
        print(f"  {k}. {r['comparison']} [{r['metric']}]: {f.format(r['mean'])} [{f.format(r['lo'])}, "
              f"{f.format(r['hi'])}], Wilcoxon p {r['p']:.1e}, Holm p {r['holm_p']:.1e}, "
              f"A better {int(r['better'])}/{int(r['n'])}")


def print_secondary(title, pairs, v, keys=("psnr", "lpips", "blockiness")):
    print(f"\n{title} (exploratory, uncorrected)")
    for key in keys:
        f = "{:+.2f}" if key == "psnr" else "{:+.3f}"
        for a, b in pairs:
            r = paired_stats(v(a, key) - v(b, key), key)
            print(f"  {key:10s} {a} − {b}: {f.format(r['mean'])} [{f.format(r['lo'])}, {f.format(r['hi'])}], "
                  f"p {r['p']:.1e}, A better {r['better']}/{r['n']}")


def print_main_table(title, methods, v):
    print(f"\n{title}")
    print(f"  {'method':36s} {'PSNR':>6s} {'PSNR(MSE)':>9s} {'LPIPS':>6s} {'block.':>6s} {'LR loss':>8s}  "
          + " ".join(f"{b:>9s}" for b in BAND_NAMES))
    for m in methods:
        p = v(m, "psnr")
        print(f"  {m:36s} {p.mean():6.2f} {psnr_of_mean_mse(p):9.2f} {v(m, 'lpips').mean():6.3f} "
              f"{v(m, 'blockiness').mean():6.3f} {v(m, 'lr_loss').mean():8.2f}  "
              + " ".join(f"{v(m, None, b).mean():9.2f}" for b in BAND_NAMES))


def write_json(path, obj):
    atomic_write_bytes(path, json.dumps(obj, indent=1).encode("utf-8"))


# ---------------------------------------------------------------- inputs
class Inputs:
    def __init__(self, num_images):
        self.index_rows = data.load_index()
        self.labels = data.labels_of(self.index_rows)
        self.lr_all = data.load_measurements()
        _, self.test_indices = data.load_split(self.labels)
        self.sample = [int(i) for i in data.get_test_sample(self.labels, self.test_indices)]
        self.images = self.sample[:num_images]
        self.best = self._csgm_best()

    def hr(self, i):
        return data.hr_image(self.index_rows, i)

    def _csgm_best(self):
        """Validate the CSGM test run and pick, per image and condition, the restart with the lowest final LR loss."""
        plan = json.loads((paths.DATA_DIR / f"{CSGM_STEM}.manifest.json").read_text())["run_plan"]
        rows = load_results(paths.DATA_DIR / f"{CSGM_STEM}.jsonl")
        key = lambda r: (r["image_index"], r["condition"], r["restart"])
        plan_by_key = {key(r): r for r in plan}
        assert len({key(r) for r in rows}) == len(rows) == len(plan), "duplicate or missing CSGM runs"
        for r in rows:
            p = plan_by_key[key(r)]
            for field in ("learning_rate", "num_steps", "num_restarts", "class_label", "true_label"):
                assert r[field] == p[field], (key(r), field)
            assert math.isfinite(r["final_loss"]) and math.isfinite(r["psnr"])
        planned_images = list(dict.fromkeys(r["image_index"] for r in plan))
        assert planned_images == self.sample, "CSGM plan order differs from the fixed test sample"
        assert np.all(np.isin(planned_images, self.test_indices))
        best = {}
        for i in self.images:
            for c in CONDITIONS:
                best[(i, c)] = min((r for r in rows if r["image_index"] == i and r["condition"] == c),
                                   key=lambda r: (r["final_loss"], r["restart"]))
        return best

    def csgm_x(self, i, c):
        with np.load(paths.DATA_DIR / f"{CSGM_STEM}_arrays" / self.best[(i, c)]["array_file"]) as d:
            return d["x_hat"]


# ---------------------------------------------------------------- part csgm
def part_csgm(inp, out_dir):
    images = inp.images
    i = images[0]
    print(f"[csgm] max |A_sr4(HR) - stored LR| (image {i}): "
          f"{float(np.abs(np.asarray(A_sr4(jnp.asarray(inp.hr(i))[None])[0]) - inp.lr_all[i]).max()):.1e}")
    post = {}
    for n, i in enumerate(images):
        y, hr = inp.lr_all[i], inp.hr(i)
        bic = bicubic(y)
        outputs = {"bicubic": bic, "bicubic + BP": backproject(bic, y),
                   "CSGM, correct label": inp.csgm_x(i, "oracle"), "CSGM, wrong label": inp.csgm_x(i, "wrong")}
        outputs["CSGM, correct label + BP"] = backproject(outputs["CSGM, correct label"], y)
        outputs["CSGM, wrong label + BP"] = backproject(outputs["CSGM, wrong label"], y)
        post[i] = {"hr_blockiness": blockiness(hr), **{m: evaluate(outputs[m], hr, y) for m in CSGM_METHODS}}
        save_npz(out_dir / f"img{i:05d}.npz",
                 {m.replace(", ", "_").replace(" + ", "_").replace(" ", "_"): outputs[m]
                  for m in CSGM_METHODS if m.endswith("+ BP")}, post[i])
        if (n + 1) % 10 == 0:
            print(f"[csgm] {n + 1}/{len(images)} images done", flush=True)
    write_json(out_dir / "metrics.json", {str(i): post[i] for i in images})

    d = max(abs(post[i]["CSGM, correct label"]["psnr"] - inp.best[(i, "oracle")]["psnr"]) for i in images)
    print(f"[csgm] max |PSNR here - runner PSNR| (CSGM, correct label): {d:.2e} dB")
    print(f"[csgm] max LR loss after BP: "
          f"{max(post[i][m]['lr_loss'] for i in images for m in CSGM_METHODS if m.endswith('+ BP')):.2e}")
    v = table([(lambda m: True, post)], images)
    print_main_table(f"[csgm] all methods, mean over {len(images)} test images "
                     f"(HR blockiness {np.mean([post[i]['hr_blockiness'] for i in images]):.3f})", CSGM_METHODS, v)
    if len(images) > 1:
        print_secondary("[csgm] paired comparisons", [
            ("bicubic + BP", "bicubic"), ("CSGM, correct label + BP", "CSGM, correct label"),
            ("CSGM, correct label + BP", "bicubic + BP"), ("CSGM, correct label", "bicubic"),
            ("CSGM, correct label + BP", "CSGM, wrong label + BP"), ("CSGM, wrong label + BP", "bicubic + BP"),
            ("CSGM, correct label", "CSGM, wrong label")], v)
    return post


# ---------------------------------------------------------------- part iadrift
def part_iadrift(inp, out_dir, post):
    images = inp.images
    iad_rows = load_results(paths.DATA_DIR / f"{IAD_STEM}.jsonl")
    iad_plan = json.loads((paths.DATA_DIR / f"{IAD_STEM}.manifest.json").read_text())["run_plan"]
    iad = {(r["image_index"], r["condition"]): r for r in iad_rows}
    assert len(iad) == len(iad_rows) == len(iad_plan) == 2 * len(inp.sample)
    for p in iad_plan:
        r = iad[(p["image_index"], p["condition"])]
        assert all(r[k] == p[k] for k in p), p
    for i in images:
        for c in CONDITIONS:
            assert iad[(i, c)]["class_label"] == inp.best[(i, c)]["class_label"]
            assert iad[(i, c)]["csgm_restart"] == inp.best[(i, c)]["restart"]
    print(f"[iadrift] {len(iad_rows)} runs, all in the plan; starts = the kept CSGM restarts; all outputs finite: "
          f"{all(r['step100']['finite'] and r['step1000']['finite'] for r in iad_rows)}; start check max "
          f"|regenerated CSGM - saved CSGM|: {max(r['start_max_abs_diff_vs_saved_csgm'] for r in iad_rows)}")

    iad_post = {}
    for n, i in enumerate(images):
        y, hr = inp.lr_all[i], inp.hr(i)
        metrics, bp_arrays = {}, {}
        for c in CONDITIONS:
            with np.load(paths.DATA_DIR / f"{IAD_STEM}_arrays" / iad[(i, c)]["array_file"]) as d:
                for s in CKPTS:
                    x = d[f"x_step{s}"]
                    x_bp = backproject(x, y)
                    name = f"IADrift @{s}, {COND_NAME[c]}"
                    metrics[name] = evaluate(x, hr, y)
                    metrics[name + " + BP"] = evaluate(x_bp, hr, y)
                    bp_arrays[f"bp_{c}_step{s}"] = x_bp
        iad_post[i] = metrics
        save_npz(out_dir / f"img{i:05d}.npz", bp_arrays, metrics)
        if (n + 1) % 10 == 0:
            print(f"[iadrift] {n + 1}/{len(images)} images done", flush=True)
    write_json(out_dir / "metrics.json", {str(i): iad_post[i] for i in images})

    d = max(abs(iad_post[i][f"IADrift @{s}, {COND_NAME[c]}"]["psnr"] - iad[(i, c)][f"step{s}"]["psnr"])
            for i in images for c in CONDITIONS for s in CKPTS)
    print(f"[iadrift] max |PSNR here - runner PSNR|: {d:.2e} dB")
    print(f"[iadrift] max LR loss after BP: "
          f"{max(iad_post[i][m]['lr_loss'] for i in images for m in IAD_METHODS if m.endswith('+ BP')):.2e}")

    v = table([(lambda m: m.startswith("IADrift"), iad_post), (lambda m: True, post)], images)
    all_methods = (CSGM_METHODS[:4] + [m for m in IAD_METHODS if "correct" in m] + CSGM_METHODS[4:]
                   + [m for m in IAD_METHODS if "wrong" in m])
    print_main_table(f"[iadrift] all methods, mean over {len(images)} test images", all_methods, v)
    if len(images) < 2:
        return iad_post
    tests = [(f"IADrift @{s}, correct label + BP", "CSGM, correct label + BP", key)
             for s in (1000, 100) for key in ("psnr", "lpips")]
    summary = primary(tests, v)
    write_json(out_dir / "primary.json", summary)
    print_primary(f"[iadrift] PRIMARY (protocol Section 6; {len(images)} test images, correct label, Holm over 4)", summary)
    print_secondary("[iadrift] secondary", [(f"IADrift @{s}, correct label", "CSGM, correct label") for s in CKPTS]
                    + [(f"IADrift @{s}, wrong label + BP", "CSGM, wrong label + BP") for s in CKPTS]
                    + [(f"IADrift @{s}, correct label + BP", "bicubic + BP") for s in CKPTS]
                    + [(f"IADrift @{s}, correct label + BP", f"IADrift @{s}, wrong label + BP") for s in CKPTS]
                    + [("IADrift @1000, correct label + BP", "IADrift @100, correct label + BP")], v)
    return iad_post


# ---------------------------------------------------------------- part ensemble
def part_ensemble(inp, out_dir, csgm_dir, csgm_post):
    images = inp.images
    run_dir = paths.DATA_DIR / ENS_RUN_DIR
    manifest = json.loads((run_dir / "manifest.json").read_text())
    spec = {k: v for k, v in manifest.items() if k != "hash"}
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()[:12]
    assert manifest["hash"] == spec_hash == ENS_HASH, (manifest["hash"], spec_hash)
    ens_rows = load_results(run_dir / f"results_{ENS_HASH}.jsonl")
    plan_keys = {(r["image_index"], *r["shift"]): r for r in manifest["run_plan"]}
    ens = {}
    for r in ens_rows:
        p = plan_keys[(r["image_index"], *r["shift"])]
        for f in ("learning_rate", "num_steps", "class_label", "start_restart"):
            assert r[f] == p[f], (r["image_index"], r["shift"], f)
        assert math.isfinite(r["final_loss"]) and math.isfinite(r["lr_loss_unshifted"])
        ens[(r["image_index"], *r["shift"])] = r
    assert len(ens) == len(ens_rows) == len(plan_keys), "duplicate or missing shifted runs"
    complete = len(images) == len(inp.sample)
    print(f"[ensemble] shifted runs: {len(ens_rows)}/{len(plan_keys)}, manifest hash {ENS_HASH}")

    def load_members(i):
        out = {}
        for r in (0, 1):
            with np.load(paths.DATA_DIR / f"{CSGM_STEM}_arrays" / f"img{i:05d}_oracle_r{r}.npz") as d:
                out[f"r{r}"] = d["x_hat"]
        for sy, sx in SHIFTS:
            with np.load(run_dir / f"img{i:05d}_shift{sy}{sx}_r0.npz") as d:
                out[f"shift{sy}{sx}"] = unshift_x(d["x_hat"], sy, sx)
        return out

    def weights(L, weighted):  # same as iadrift.ensemble.weights
        w = np.exp(-(L - L.min()) / max(L.std(), 1e-6)) if weighted else np.ones(len(L))
        return w / w.sum()

    ens_post, ref_check, loss_check = {}, [], []
    for n, i in enumerate(images):
        y, hr = inp.lr_all[i], inp.hr(i)
        members = load_members(i)
        L = {m: lr_loss(x, y) for m, x in members.items()}
        kept = inp.best[(i, "oracle")]["restart"]  # lowest final LR loss
        outputs, info = {}, {"L": L, "kept": kept}
        outputs["CSGM, correct label + BP"] = backproject(members[f"r{kept}"], y)
        for name, names in ENSEMBLES.items():
            w = weights(np.array([L[m] for m in names]), name.startswith("Ens-W"))
            mean_x = sum(wk * members[m] for wk, m in zip(w, names))
            info[name] = {"weights": w.tolist(), "n_eff": float(1 / np.sum(w ** 2)),
                          "mse_raw": mse(hr, mean_x), "lr_loss_raw": lr_loss(mean_x, y)}
            outputs[name + " + BP"] = backproject(mean_x, y)
        info["member_mse"] = {m: mse(hr, x) for m, x in members.items()}
        ens_post[i] = {"info": info, **{m: evaluate(x, hr, y) for m, x in outputs.items()}}
        _, ref = load_npz(csgm_dir / f"img{i:05d}.npz")
        ref_check.append(float(np.abs(ref["CSGM_correct_label_BP"] - outputs["CSGM, correct label + BP"]).max()))
        loss_check += [abs(L[f"shift{sy}{sx}"] - ens[(i, sy, sx)]["lr_loss_unshifted"]) / L[f"shift{sy}{sx}"]
                       for sy, sx in SHIFTS]
        save_npz(out_dir / f"img{i:05d}.npz", {**{m.split(":")[0] + "_BP": outputs[m] for m in ENS_METHODS},
                                               "CSGM_BP": outputs["CSGM, correct label + BP"]}, ens_post[i])
        if (n + 1) % 10 == 0:
            print(f"[ensemble] {n + 1}/{len(images)} images done", flush=True)
    write_json(out_dir / "metrics.json", {str(i): ens_post[i] for i in images})

    print(f"[ensemble] max |CSGM + BP here - {csgm_dir.name}| (pixels): {max(ref_check):.2e}")
    print(f"[ensemble] max relative |L (CPU, here) - L (GPU, runner)| of shifted members: {max(loss_check):.2e}")
    d = max(abs(ens_post[i]["CSGM, correct label + BP"][k] - csgm_post[str(i)]["CSGM, correct label + BP"][k])
            for i in images for k in ("psnr", "lpips"))
    print(f"[ensemble] max |CSGM + BP metrics here - CSGM part| (PSNR dB / LPIPS): {d:.2e}")
    print(f"[ensemble] max LR loss after BP: {max(ens_post[i][m]['lr_loss'] for i in images for m in ENS_METHODS):.2e}")

    v = table([(lambda m: m.startswith("bicubic"), {i: csgm_post[str(i)] for i in images}),
               (lambda m: True, ens_post)], images)
    print_main_table(f"[ensemble] all methods (+ BP), mean over {len(images)} test images, correct label",
                     ["bicubic + BP", "CSGM, correct label + BP"] + ENS_METHODS, v)
    if len(images) < 2:
        return ens_post
    summary = primary([("Ens-W: weighted 5 CSGM runs + BP", "CSGM, correct label + BP", key)
                       for key in ("psnr", "lpips")], v)
    if complete:  # primary.json only for the full test sample
        write_json(out_dir / "primary.json", summary)
    print_primary(f"[ensemble] PRIMARY (protocol Section 11; {len(images)} test images, Holm over 2)", summary)
    print_secondary("[ensemble] secondary", [(m, "CSGM, correct label + BP") for m in ENS_METHODS
                                             if not m.startswith("Ens-W")]
                    + [(m, "bicubic + BP") for m in ENS_METHODS]
                    + [("Ens-W: weighted 5 CSGM runs + BP", "Ens-U: mean of 5 CSGM runs + BP")], v)
    n_eff = np.array([ens_post[i]["info"]["Ens-W: weighted 5 CSGM runs"]["n_eff"] for i in images])
    gain = np.array([10 * np.log10(np.mean(list(ens_post[i]["info"]["member_mse"].values()))
                                   / ens_post[i]["info"]["Ens-U: mean of 5 CSGM runs"]["mse_raw"]) for i in images])
    half = stats.t.ppf(0.975, len(gain) - 1) * gain.std(ddof=1) / np.sqrt(len(gain))
    print(f"\n[ensemble] Ens-W effective members 1/sum w^2: mean {n_eff.mean():.2f} (min {n_eff.min():.2f}, "
          f"max {n_eff.max():.2f}) of 5")
    print(f"[ensemble] MMSE check (Ens-U, before BP): mean gain {gain.mean():.2f} dB [95% CI {gain.mean() - half:.2f}, "
          f"{gain.mean() + half:.2f}]; independent posterior samples would give {10 * np.log10(2 / 1.2):.2f} dB")
    return ens_post


# ---------------------------------------------------------------- comparison with the caches (--check)
def compare_values(new, old, path, stats_, rtol):
    if isinstance(old, dict):
        if not isinstance(new, dict) or set(new) != set(old):
            stats_["structure"].append(path)
            return
        for k in old:
            compare_values(new[k], old[k], f"{path}/{k}", stats_, rtol)
    elif isinstance(old, list):
        if not isinstance(new, list) or len(new) != len(old):
            stats_["structure"].append(path)
            return
        for k, (a, b) in enumerate(zip(new, old)):
            compare_values(a, b, f"{path}[{k}]", stats_, rtol)
    elif isinstance(old, (int, float)) and not isinstance(old, bool):
        stats_["n"] += 1
        a, b = float(new), float(old)
        if a == b:
            stats_["exact"] += 1
            return
        rel = abs(a - b) / max(abs(b), 1e-300)
        if rel > stats_["max_rel"]:
            stats_["max_rel"], stats_["worst"] = rel, (path, a, b)
        if rel > rtol:
            stats_["over_tol"].append((path, a, b, rel))
    else:
        stats_["n"] += 1
        if new == old:
            stats_["exact"] += 1
        else:
            stats_["over_tol"].append((path, new, old, None))


def report_values(label, new, old, rtol):
    s = {"n": 0, "exact": 0, "max_rel": 0.0, "worst": None, "over_tol": [], "structure": []}
    compare_values(new, old, "", s, rtol)
    ok = not s["over_tol"] and not s["structure"]
    print(f"  {label}: {s['n']} values, {s['exact']} identical, max relative difference {s['max_rel']:.2e}"
          + (f" at {s['worst'][0]}" if s["worst"] else "") + f"; > {rtol:g}: {len(s['over_tol'])}"
          + (f"; structure mismatches: {s['structure'][:3]}" if s["structure"] else "")
          + ("" if ok else "  <-- DIFFERS"))
    for path, a, b, rel in s["over_tol"][:5]:
        print(f"      {path}: new {a!r} vs cache {b!r}" + (f" (rel {rel:.2e})" if rel is not None else ""))
    return ok


def check_part(part, new_dir, old_dir, images, full, rtol):
    print(f"\n[check {part}] {new_dir} vs {old_dir}")
    ok = True
    new_m = json.loads((new_dir / "metrics.json").read_text())
    old_m = json.loads((old_dir / "metrics.json").read_text())
    assert list(new_m) == [str(i) for i in images]
    ok &= report_values(f"metrics.json ({len(images)} images)", new_m, {str(i): old_m[str(i)] for i in images}, rtol)
    if full:
        ok &= list(new_m) == list(old_m)
    npz_values = {"n": 0, "identical": 0, "max_abs": 0.0, "names": None}
    metrics_new, metrics_old = {}, {}
    for i in images:
        m_new, a_new = load_npz(new_dir / f"img{i:05d}.npz")
        m_old, a_old = load_npz(old_dir / f"img{i:05d}.npz")
        metrics_new[str(i)], metrics_old[str(i)] = m_new, m_old
        if set(a_new) != set(a_old):
            print(f"  img{i:05d}.npz: array names differ: {sorted(a_new)} vs {sorted(a_old)}  <-- DIFFERS")
            ok = False
            continue
        npz_values["names"] = sorted(a_new)
        for k in a_old:
            npz_values["n"] += 1
            same = a_new[k].dtype == a_old[k].dtype and np.array_equal(a_new[k], a_old[k])
            npz_values["identical"] += same
            if not same:
                npz_values["max_abs"] = max(npz_values["max_abs"], float(np.abs(
                    a_new[k].astype(np.float64) - a_old[k].astype(np.float64)).max()))
    print(f"  per-image .npz arrays {npz_values['names']}: {npz_values['identical']}/{npz_values['n']} identical, "
          f"max abs difference {npz_values['max_abs']:.2e}")
    ok &= npz_values["identical"] == npz_values["n"]
    ok &= report_values("per-image .npz metrics_json", metrics_new, metrics_old, rtol)
    if (old_dir / "primary.json").exists():
        if full:
            ok &= report_values("primary.json", json.loads((new_dir / "primary.json").read_text()),
                                json.loads((old_dir / "primary.json").read_text()), rtol)
        else:
            print("  primary.json: not compared (computed on a subset of the images)")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("csgm", "iadrift", "ensemble", "all"), default="all")
    parser.add_argument("--num-images", type=int, default=data.TEST_SAMPLE_SIZE,
                        help="the first N images of the fixed test sample (primary tests are stored for any N >= 2; "
                             "the ensemble's primary.json only for N = 100)")
    parser.add_argument("--out", type=Path, default=None,
                        help="parent directory of the part directories (default: IADRIFT_DATA, i.e. the saved caches)")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into an existing cache directory")
    parser.add_argument("--check", action="store_true",
                        help="recompute into a temporary directory and compare with the caches in IADRIFT_DATA")
    parser.add_argument("--rtol", type=float, default=1e-9, help="--check: relative tolerance for stored metrics")
    parser.add_argument("--keep", action="store_true", help="--check: keep the temporary directory")
    args = parser.parse_args()
    if not 1 <= args.num_images <= data.TEST_SAMPLE_SIZE:
        parser.error(f"--num-images must be between 1 and {data.TEST_SAMPLE_SIZE}")
    if args.check and args.out is not None:
        parser.error("--check writes to a temporary directory; do not combine it with --out")

    import torch
    torch.set_num_threads(4)  # LPIPS on the CPU; the caches were computed with 4 threads

    parts = ("csgm", "iadrift", "ensemble") if args.part == "all" else (args.part,)
    out = Path(tempfile.mkdtemp(prefix="evaluate_test_check_")) if args.check else (args.out or paths.DATA_DIR)
    out_dirs = {p: out / PART_DIRS[p] for p in parts}
    for p, d in out_dirs.items():
        if d.exists() and any(d.iterdir()) and not args.overwrite:
            raise SystemExit(f"{d} exists and is not empty (an existing cache); use --overwrite to replace it, "
                             f"--out DIR to write elsewhere, or --check to compare with it.")
        d.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out}" + (" (temporary, --check)" if args.check else ""))

    inp = Inputs(args.num_images)
    print(f"Test images: {len(inp.images)} of {len(inp.sample)} (first {inp.images[:4]})")
    post = None
    if "csgm" in parts:
        post = part_csgm(inp, out_dirs["csgm"])
    csgm_dir = out / PART_DIRS["csgm"]
    if not (csgm_dir / "metrics.json").exists():
        csgm_dir = paths.DATA_DIR / PART_DIRS["csgm"]
    if post is None and ("iadrift" in parts or "ensemble" in parts):
        print(f"CSGM part read from {csgm_dir}")
        loaded = json.loads((csgm_dir / "metrics.json").read_text())
        post = {i: loaded[str(i)] for i in inp.images}
    if "iadrift" in parts:
        part_iadrift(inp, out_dirs["iadrift"], post)
    if "ensemble" in parts:
        part_ensemble(inp, out_dirs["ensemble"], csgm_dir, {str(i): post[i] for i in inp.images})

    if args.check:
        full = len(inp.images) == len(inp.sample)
        ok = all([check_part(p, out_dirs[p], paths.DATA_DIR / PART_DIRS[p], inp.images, full, args.rtol)
                  for p in parts])
        if args.keep:
            print(f"Kept {out}")
        else:
            shutil.rmtree(out)
        print("\nCHECK PASSED" if ok else "\nCHECK FAILED")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
