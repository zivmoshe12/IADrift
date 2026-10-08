"""Export the development ablations behind notebook 04 ("development ablations") into results/ (CPU).

Development images only (never test images), correct label. Reads cached development outputs and recomputes every
metric on the CPU (iadrift.metrics, iadrift.operators):

    results/ablation_metrics.json   per section: per-image metrics (PSNR, LPIPS, LR loss, band SNR, blockiness), and the
                                    section-specific numbers (variance shares, eps statistics, screening runs, weights)
    results/ablation_examples.npz   images for the figures (float16; the z x eps grid as uint8 at 128x128)

Sections and sources (IADRIFT_DATA), with the runners that made them:
    1 z vs eps            noise_label_role_dev_v1/        scripts/dev/run_noise_label_role_dev.py
    2 image in eps        image_in_eps_dev_v1.npz         scripts/dev/run_image_in_eps_dev.py
    3 eps on the sphere   csgm_dev_5000steps_lr0.02/      scripts/dev/run_csgm_dev.py
                          csgm_eps_projection_dev_v1/     scripts/dev/run_eps_projection_dev.py
    4 noise-label search  noise_label_search_dev_v1/      scripts/dev/run_noise_label_search_dev.py
    5 ensembles           csgm_dev_5000steps_lr0.02/, z_swap_dev_v1/confirm/ (run_z_swap_confirm_dev.py),
                          ensemble_dev_v1/ and one_image_demos_v1/D_shift##_r0.npz (run_ensemble_dev.py),
                          iadrift_ensemble_dev_v1/ (run_iadrift_ensemble_dev.py)
    6 colour penalty and frequency anchor   hf_constraints_dev_v1/ (run_hf_constraints_dev.py) + the section-5
                          members of image 403
Ensemble weights: w_k ~ exp(-(L_k - L_min) / std L) from the CSGM members' LR losses (iadrift.ensemble.weights; never
HR); the CSGM restart kept by the usual rule = the lower LR loss of r0, r1.

Ported from github/scripts/export_ablation_results.py (helpers from export_dev_results.py / freq_metrics.py, now
iadrift.metrics and iadrift.operators).

Usage:
    python scripts/analysis/export_ablation_results.py            # write results/ablation_{metrics.json,examples.npz}
    python scripts/analysis/export_ablation_results.py --check    # export to a temporary dir, compare with results/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

from iadrift.runtime import setup_cpu

setup_cpu()

import argparse
import json
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import torch

from iadrift import data, paths
from iadrift.ensemble import weights
from iadrift.metrics import LR_NYQUIST, band_filter, band_snr_db, blockiness, lpips_dist, psnr
from iadrift.operators import A_sr4, backproject, bicubic_up

DATA_DIR = paths.DATA_DIR
DEV4 = list(data.DEV_IMAGES_4)
DEV6 = list(data.DEV_IMAGES_6)
CSGM_DEV = DATA_DIR / "csgm_dev_5000steps_lr0.02"
MEMBERS = {"r0": (0, 0), "r1": (0, 0), "shift01": (0, 1), "shift10": (1, 0), "shift11": (1, 1)}
FILES = ("ablation_metrics.json", "ablation_examples.npz")
EX = {}

torch.set_num_threads(4)  # as in the original export (LPIPS on the CPU)


def lr_loss(x, y):
    """||y - A x||^2 for x [256, 256, 3], y [64, 64, 3] (the export's own form: [0] before the subtraction)."""
    return float(jnp.sum((A_sr4(jnp.asarray(x, jnp.float32)[None])[0] - jnp.asarray(y, jnp.float32)) ** 2))


def evaluate(x, hr, y):
    return {"psnr": psnr(hr, x), "lpips": lpips_dist(x, hr), "lr_loss": lr_loss(x, y),
            "band_snr_db": band_snr_db(x, hr), "blockiness": blockiness(x)}


def with_bp(x, hr, y):
    return {"raw": evaluate(x, hr, y), "bp": evaluate(backproject(x, y), hr, y)}


def load(path, key="x_hat"):
    with np.load(path) as d:
        return d[key], (json.loads(str(d["metrics_json"])) if "metrics_json" in d.files else {})


def keep(name, x, dtype=np.float16):
    EX[name] = np.asarray(x, dtype)


def unshift(x, m):
    sy, sx = MEMBERS[m]
    return np.roll(x, (-4 * sy, -4 * sx), axis=(0, 1))


def init_eps(i, r):
    """The CSGM start eps of image i, restart r (= Generator.initialize, eps part only; no model needed)."""
    key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(data.SWEEP_SEED), i), r)
    return np.asarray(jax.random.normal(jax.random.split(key)[0], (1, 256, 256, 3), dtype=jnp.float32))


def eps_stats(e, e0=None):
    p = np.asarray(e, np.float64).reshape(16, 16, 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(256, 768)
    pn = np.linalg.norm(p, axis=1) / np.sqrt(768)
    s = {"rms": float(np.sqrt(np.mean(p ** 2))), "patch_norm_mean": float(pn.mean()), "patch_norm_std": float(pn.std()),
         "patch_norm_max": float(pn.max())}
    if e0 is not None:
        p0 = np.asarray(e0, np.float64).reshape(16, 16, 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(256, 768)
        s["rel_change"] = float(np.linalg.norm(p - p0) / np.linalg.norm(p0))
    return s


def restart_path(i, r):
    """CSGM restart member: the development CSGM runs (4 images) or the Check 10b CSGM runs (9780, 8405)."""
    if i in DEV4:
        return CSGM_DEV / f"img{i:05d}_oracle_r{r}.npz"
    return DATA_DIR / "z_swap_dev_v1" / "confirm" / f"img{i:05d}_csgm_r{r}.npz"


def csgm_kept(i):
    """Saved development CSGM run with the lowest final LR loss: (x_hat, eps, restart)."""
    runs = []
    for r in (0, 1):
        with np.load(restart_path(i, r)) as d:
            runs.append((json.loads(str(d["metrics_json"]))["final_loss"], r, d["x_hat"], d["eps"]))
    _, r, x, e = min(runs, key=lambda t: (t[0], t[1]))
    return x, e, r


def chroma_hf(x):
    cb = -0.1687 * x[..., 0] - 0.3313 * x[..., 1] + 0.5 * x[..., 2]
    cr = 0.5 * x[..., 0] - 0.4187 * x[..., 1] - 0.0813 * x[..., 2]
    return float(np.sum(np.asarray(band_filter(np.stack([cb, cr], -1), LR_NYQUIST, np.inf)) ** 2))


def export(out_dir):
    EX.clear()
    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    names = data.class_names()
    _, test = data.load_split(labels)
    assert not np.any(np.isin(DEV6, test)), "development images only"
    lr_all = data.load_measurements()
    img = {i: {"hr": data.hr_image(index_rows, i), "y": np.asarray(lr_all[i], np.float32), "label": int(labels[i])}
           for i in DEV6}
    out = {"classes": {str(i): names[img[i]["label"]] for i in DEV6}}

    # ---- 1. z vs eps (generator only) ----
    role_dir = DATA_DIR / "noise_label_role_dev_v1"
    role = json.loads((role_dir / "summary.json").read_text())
    loc = {c: json.loads((role_dir / f"locality_class{c:03d}.json").read_text()) for c in role["classes"]}
    keys = {"eps_patch": ("in_tile", "in_3x3", "entropy", "rms_change"), "z_coord": ("entropy", "rms_change")}
    out["z_vs_eps"] = {"classes": role["classes"], "class_names": {str(c): names[c] for c in role["classes"]},
                       "anova_shares": role["anova_shares"],
                       "locality": {kind: {k: float(np.mean([r[k] for c in role["classes"] for r in loc[c][kind]]))
                                           for k in keys[kind]} | {"n": sum(len(loc[c][kind]) for c in role["classes"])}
                                    for kind in ("eps_patch", "z_coord")}}
    with np.load(role_dir / "grid_class134.npz") as d:
        g = d["x"].astype(np.float32)
    g = g.reshape(8, 8, 128, 2, 128, 2, 3).mean(axis=(3, 5))  # 2x2 average pooling for display
    keep("grid_crane", np.round((np.clip(g, -1, 1) + 1) * 127.5), np.uint8)
    print("1 done", flush=True)

    # ---- 2. an image in eps ----
    gen = np.load(DATA_DIR / "image_in_eps_dev_v1.npz")
    sec = {}
    for i in (403, 3939, 8405):
        sec[str(i)] = {}
        for src in ("bicubic(y)", "HR (oracle)"):
            for tag in ("a0", "a0.25", "a0.5", "a0.75", "a1", "raw"):
                x = gen[f"{i}|{src}|{tag}"]
                m = evaluate(x, img[i]["hr"], img[i]["y"])
                sec[str(i)][f"{src}|{tag}"] = {k: m[k] for k in ("psnr", "lpips", "lr_loss")}
                if i == 403 and tag in ("a0", "a0.5", "a1"):
                    keep(f"eps_img|{i}|{src}|{tag}", x)
        x_csgm, _, _ = csgm_kept(i)
        sec[str(i)]["CSGM (5,000 steps)"] = {k: v for k, v in evaluate(x_csgm, img[i]["hr"], img[i]["y"]).items()
                                             if k in ("psnr", "lpips", "lr_loss")}
    keep("eps_img|403|HR", img[403]["hr"])
    keep("eps_img|403|bicubic", bicubic_up(img[403]["y"]))
    out["image_in_eps"] = sec
    print("2 done", flush=True)

    # ---- 3. eps statistics and sphere projection ----
    sec = {}
    for i in DEV4:
        x, e, r = csgm_kept(i)
        hr, y = img[i]["hr"], img[i]["y"]
        sec[str(i)] = {"none": {**with_bp(x, hr, y), "eps": eps_stats(e, init_eps(i, r)),
                                "init_eps": eps_stats(init_eps(i, r))}}
        if i == 403:
            keep("sphere|403|none", x)
        for slug, label in (("global_sphere", "global sphere"), ("perpatch_sphere", "per-patch sphere")):
            runs = [load(DATA_DIR / "csgm_eps_projection_dev_v1" / f"img{i:05d}_{slug}_r{rr}.npz") for rr in (0, 1)]
            rr = min((0, 1), key=lambda k: (runs[k][1]["final_loss"], k))
            with np.load(DATA_DIR / "csgm_eps_projection_dev_v1" / f"img{i:05d}_{slug}_r{rr}.npz") as d:
                e = d["eps"]
            sec[str(i)][label] = {**with_bp(runs[rr][0], hr, y), "eps": eps_stats(e, init_eps(i, rr))}
            if i == 403:
                keep(f"sphere|403|{label}", runs[rr][0])
    keep("sphere|403|HR", img[403]["hr"])
    out["sphere"] = sec
    print("3 done", flush=True)

    # ---- 4. noise-label search ----
    nls_dir = DATA_DIR / "noise_label_search_dev_v1"
    summ = json.loads((nls_dir / "summary.json").read_text())
    sec = {}
    for i in DEV4:
        hr, y = img[i]["hr"], img[i]["y"]
        s = summ[str(i)]
        x_base, _, r = csgm_kept(i)
        x_s, _ = load(nls_dir / f"final_img{i:05d}_{s['best']}.npz")
        screen = []
        for key, loss in s["screen"].items():
            xs, _ = load(nls_dir / f"screen_img{i:05d}_{key}.npz")
            screen.append({"key": key, "lr_loss": loss, "psnr": psnr(hr, xs), "lpips": lpips_dist(xs, hr)})
        sec[str(i)] = {"best": s["best"], "baseline": with_bp(x_base, hr, y), "search": with_bp(x_s, hr, y),
                       "screen": screen}
        if i == 3939:
            for k, x in (("baseline", x_base), ("search", x_s)):
                keep(f"nls|{i}|{k}", x)
                keep(f"nls|{i}|{k} + BP", backproject(x, y))
            keep(f"nls|{i}|HR", hr)
    out["noise_label_search"] = sec
    print("4 done", flush=True)

    # ---- 5. ensembles of CSGM and of IADrift runs ----
    sec = {}
    for i in DEV6:
        hr, y = img[i]["hr"], img[i]["y"]
        csgm = {}
        for m in MEMBERS:
            if m in ("r0", "r1"):
                path = restart_path(i, int(m[1]))
            elif i == 403:
                path = DATA_DIR / "one_image_demos_v1" / f"D_{m}_r0.npz"
            else:
                path = DATA_DIR / "ensemble_dev_v1" / f"img{i:05d}_{m}_r0.npz"
            csgm[m] = unshift(load(path)[0], m)
        iad = {ck: {m: unshift(load(DATA_DIR / "iadrift_ensemble_dev_v1" / f"img{i:05d}_{m}.npz", f"x_step{ck}")[0], m)
                    for m in MEMBERS} for ck in (100, 1000)}
        L = [lr_loss(csgm[m], y) for m in MEMBERS]
        w = weights(L, weighted=True)
        kept = min(("r0", "r1"), key=lambda m: (L[list(MEMBERS).index(m)], m))
        outs = {"CSGM": csgm[kept], "CSGM E3": np.mean(list(csgm.values()), axis=0),
                "CSGM E3w": sum(wk * csgm[m] for wk, m in zip(w, MEMBERS))}
        for ck in (100, 1000):
            outs[f"IADrift @{ck}"] = iad[ck][kept]
            outs[f"IADrift @{ck} E3"] = np.mean(list(iad[ck].values()), axis=0)
            outs[f"IADrift @{ck} E3w"] = sum(wk * iad[ck][m] for wk, m in zip(w, MEMBERS))
        bp = {n: backproject(x, y) for n, x in outs.items()}
        sec[str(i)] = {"weights": dict(zip(MEMBERS, w.tolist())), "kept": kept, "L": dict(zip(MEMBERS, L)),
                       **{n: evaluate(x, hr, y) for n, x in bp.items()}}
        if i == 403:
            keep("ens|403|HR", hr)
            for n in ("CSGM", "CSGM E3", "IADrift @1000", "IADrift @1000 E3", "IADrift @100 E3"):
                keep(f"ens|403|{n} + BP", bp[n])
            members_403, w_403 = (csgm, iad), w
        print(f"5: image {i} done", flush=True)
    out["ensembles"] = sec

    # ---- 6. colour penalty (A) and frequency anchor (B), image 403 ----
    i, hfc = 403, DATA_DIR / "hf_constraints_dev_v1"
    hr, y = img[i]["hr"], img[i]["y"]
    csgm, iad = members_403
    A = {m: unshift(load(hfc / f"A_lam0.1_{m}.npz")[0], m) for m in MEMBERS}
    wa = weights([load(hfc / f"A_lam0.1_{m}.npz")[1]["fit_loss"] for m in MEMBERS], weighted=True)
    B = {ck: {m: unshift(load(hfc / f"B_from_base_{m}.npz", f"x_step{ck}")[0], m) for m in MEMBERS}
         for ck in (100, 1000)}
    AB = {ck: {m: unshift(load(hfc / f"B_from_A_{m}.npz", f"x_step{ck}")[0], m) for m in MEMBERS}
          for ck in (100, 1000)}
    variants = {"CSGM": (csgm, w_403), "CSGM + A": (A, wa),
                "IADrift @100": (iad[100], w_403), "IADrift @1000": (iad[1000], w_403),
                "IADrift + B @100": (B[100], w_403), "IADrift + B @1000": (B[1000], w_403),
                "A → IADrift + B @100": (AB[100], wa), "A → IADrift + B @1000": (AB[1000], wa)}
    sec = {}
    for n, (mem, w) in variants.items():
        for kind, x in (("single", mem["r0"]), ("uniform", np.mean([mem[m] for m in MEMBERS], axis=0)),
                        ("weighted", sum(wk * mem[m] for wk, m in zip(w, MEMBERS)))):
            xb = backproject(x, y)
            sec[f"{n}|{kind}"] = {**evaluate(xb, hr, y), "chroma_hf": chroma_hf(xb)}
            if kind in ("single", "weighted") and n in ("CSGM", "CSGM + A", "IADrift @1000", "IADrift + B @1000",
                                                         "A → IADrift + B @1000"):
                keep(f"hfc|{n}|{kind}", xb)
    x = backproject(bicubic_up(y), y)
    sec["bicubic + BP|single"] = {**evaluate(x, hr, y), "chroma_hf": chroma_hf(x)}
    keep("hfc|HR", hr)
    out["hf_constraints"] = sec
    print("6 done", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_metrics.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(out_dir / "ablation_examples.npz", **EX)
    for f in FILES:
        print(f"{f:24s} {(out_dir / f).stat().st_size / 1e6:6.2f} MB")


# ---------------------------------------------------------------- --check
def max_rel_diff(a, b, path=""):
    """(largest |a - b| / max(|b|, 1), where); keys, strings and list lengths must match exactly."""
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, set(a) ^ set(b))
        return max((max_rel_diff(a[k], b[k], f"{path}/{k}") for k in a), default=(0.0, path))
    if isinstance(a, list):
        assert len(a) == len(b), path
        return max((max_rel_diff(u, v, f"{path}[{n}]") for n, (u, v) in enumerate(zip(a, b))), default=(0.0, path))
    if isinstance(a, str):
        assert a == b, (path, a, b)
        return 0.0, path
    return abs(float(a) - float(b)) / max(abs(float(b)), 1.0), path


def check(fresh_dir, ref_dir=paths.RESULTS_DIR):
    """Compare a fresh export with results/. The saved files reproduce bit for bit only on the platform they were made
    on: on the CPU even the original export's bicubic_up differs from the saved image by one float16 step, so the
    comparison uses tolerances (relative 1e-4 for the metrics, 2e-3 for the float16 images) plus the exact choices."""
    fresh = json.loads((fresh_dir / FILES[0]).read_text())
    ref = json.loads((ref_dir / FILES[0]).read_text())
    diff, where = max_rel_diff(fresh, ref)
    choices = all(fresh["ensembles"][k]["kept"] == ref["ensembles"][k]["kept"] for k in ref["ensembles"]) and all(
        fresh["noise_label_search"][k]["best"] == ref["noise_label_search"][k]["best"]
        for k in ref["noise_label_search"])
    print(f"{FILES[0]}: identical {fresh == ref}; max relative difference {diff:.1e} at {where}; same choices {choices}")
    with np.load(fresh_dir / FILES[1]) as a, np.load(ref_dir / FILES[1]) as b:
        same_keys = set(a.files) == set(b.files)
        unequal = [k for k in b.files if k in a.files and not np.array_equal(a[k], b[k])]
        worst = max((float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) for k in unequal),
                    default=0.0)
    print(f"{FILES[1]}: same arrays {same_keys}; {len(b.files) - len(unequal)} / {len(b.files)} bit-identical"
          + (f"; the others differ by at most {worst:.1e}" if unequal else ""))
    ok = same_keys and choices and diff < 1e-4 and worst < 2e-3
    print("CHECK PASSED" if ok else "CHECK FAILED")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="export into a temporary directory and compare with results/ (writes nothing there)")
    parser.add_argument("--out", type=Path, default=paths.RESULTS_DIR, help="output directory (default: results/)")
    args = parser.parse_args()
    if not args.check:
        export(args.out)
        return
    with tempfile.TemporaryDirectory(prefix="export_ablation_check_") as tmp:
        export(Path(tmp))
        raise SystemExit(0 if check(Path(tmp)) else 1)


if __name__ == "__main__":
    main()
