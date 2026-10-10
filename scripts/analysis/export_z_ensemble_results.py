"""Analysis of the larger z-ensembles (scripts/dev/run_z_ensemble_dev.py) -> results/ and figures/ (CPU).

Development only (6 images, correct label), post-hoc; never run on the test set. HR is used only for scoring; ensemble
weights come from the members' LR losses at the same step budget T (iadrift.ensemble.weights).

For each image, budget T in {500, 1000, 2000, 5000, iad100} and K in {1, 2, 4, 5, 8, 12}: N_SUBSETS random K-subsets
of the 12 arm-A members r0..11 (fixed seed, the same subsets for every T and image, 1 subset at K = 12), uniform and
weighted mean, back-projection, and scores before and after BP (PSNR, LPIPS, blockiness, band SNR). Also
n_eff = 1 / sum w^2 and the MMSE gain 10 log10(mean member MSE / ensemble MSE) before BP, against the prediction for
independent posterior samples, 10 log10(2 / (1 + 1/K)). Budget iad100 is each member's CSGM solution at step 5,000
followed by IADrift @100; it is weighted by the IADrift LR losses at step 100 ("weighted") or by the CSGM losses at
step 5,000 ("weighted_csgm", as in the IADrift ensembles of notebook 04, Section 5).

Reported:
    k_curve          PSNR / LPIPS against K for every budget
    matched_compute  12 x 500 / 1000 / 2000 / 5000, 5 x 5000, 5 and 12 x (5000 + IADrift @100), against bicubic + BP
                     (computed here) and the 5-member ensembles of notebook 04, Section 5 (results/ablation_metrics.json,
                     "ensembles"; CSGM, Ens-U, Ens-W, and the same for IADrift @100)
    arm_c            z only (4 z), grid shifts only (1 z x 4 phases), both (2 z x 4 phases, 4 z x 4 phases), and the
                     arm-A z-only ensembles at K = 4, 8, 12 for comparison at equal K
Per-image results are cached in IADRIFT_DATA/z_ensemble_dev_v1/analysis/img#####.json and recomputed when the member
files change.

Usage:
    python scripts/analysis/export_z_ensemble_results.py                   # every image with arm-A runs
    python scripts/analysis/export_z_ensemble_results.py --images 403      # a subset
Outputs: results/z_ensemble_dev.json, figures/z_ensemble_dev.png (PSNR and LPIPS against K),
         figures/z_ensemble_dev_examples.png (reconstructions vs bicubic, full images and zoomed crops)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

from iadrift.runtime import setup_cpu

setup_cpu()

import argparse
import itertools
import json

import numpy as np
import torch

from iadrift import data, paths
from iadrift.checkpoint import atomic_write_bytes, load_npz
from iadrift.ensemble import SHIFTS, effective_members, weights
from iadrift.metrics import band_snr_db, blockiness, lpips_dist, psnr
from iadrift.operators import backproject, bicubic_up

RUN_DIR = paths.DATA_DIR / "z_ensemble_dev_v1"
CACHE_DIR = RUN_DIR / "analysis"
OUT_JSON = paths.RESULTS_DIR / "z_ensemble_dev.json"
OUT_PNG = paths.FIGURES_DIR / "z_ensemble_dev.png"
OUT_EXAMPLES = paths.FIGURES_DIR / "z_ensemble_dev_examples.png"
IMAGES = list(data.DEV_IMAGES_6)
STEPS = (500, 1_000, 2_000, 5_000)
IAD = "iad100"  # budget key: CSGM 5,000 steps + IADrift @100
BUDGETS = STEPS + (IAD,)
N_MEMBERS = 12
KS = (1, 2, 4, 5, 8, 12)
N_SUBSETS = 20
SUBSET_SEED = 2026
ARM_C_STEPS = (2_000, 5_000)
BANDS = ("low", "mid", "high", "very high")
MATCHED = (("12 x 500", 12, 500), ("12 x 1000", 12, 1_000), ("12 x 2000", 12, 2_000), ("5 x 5000", 5, 5_000),
           ("12 x 5000", 12, 5_000), ("5 x (5000 + IADrift 100)", 5, IAD), ("12 x (5000 + IADrift 100)", 12, IAD))
REFS = ("bicubic", "CSGM", "CSGM Ens-U", "CSGM Ens-W", "IADrift @100", "IADrift @100 Ens-U", "IADrift @100 Ens-W")

torch.set_num_threads(4)


def modes(t):
    return ("uniform", "weighted", "weighted_csgm") if t == IAD else ("uniform", "weighted")


def total_steps(k, t):
    return k * (5_100 if t == IAD else t)


def predicted_gain_db(k):
    return float(10 * np.log10(2 / (1 + 1 / k)))


def subsets(n, k):
    """N_SUBSETS random k-subsets of range(n) (one when k == n), fixed seed per k (and per n if n < N_MEMBERS)."""
    if k == n:
        return [tuple(range(n))]
    rng = np.random.default_rng([SUBSET_SEED, k] if n == N_MEMBERS else [SUBSET_SEED, k, n])
    return [tuple(sorted(rng.choice(n, k, replace=False).tolist())) for _ in range(N_SUBSETS)]


def mse(a, b):
    return float(np.mean((np.asarray(a, np.float64) - b) ** 2))


def score(x, hr, y):
    """PSNR before BP; PSNR, LPIPS, blockiness and band SNR after BP."""
    b = backproject(x, y)
    return {"psnr_raw": psnr(hr, x), "psnr": psnr(hr, b), "lpips": lpips_dist(b, hr), "blockiness": blockiness(b),
            "band_snr_db": band_snr_db(b, hr)}


def combine(xs, losses, weighted, hr, y):
    """(Weighted) mean of the members xs [K, 256, 256, 3], scored, with n_eff and the MMSE gain (before BP)."""
    w = weights(losses, weighted)
    x = np.tensordot(w, xs, axes=1)
    m = score(x, hr, y)
    m["n_eff"] = effective_members(w)
    m["mmse_gain_db"] = float(10 * np.log10(np.mean([mse(xk, hr) for xk in xs]) / mse(x, hr)))
    return m


def mean_metrics(rows):
    """Average a list of metric dicts (band SNR per band)."""
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "band_snr_db"}
    out["band_snr_db"] = {b: float(np.mean([r["band_snr_db"][b] for r in rows])) for b in BANDS}
    return out


def ensembles_over(sets, xs_by_t, ls_by_t, hr, y, steps):
    """{T: {mode: averaged metrics}} over a list of index sets into the member arrays."""
    res = {}
    for t in steps:
        res[str(t)] = {}
        for mode in modes(t):
            ls = ls_by_t[5_000] if mode == "weighted_csgm" else ls_by_t[t]
            rows = [combine(xs_by_t[t][list(s)], ls[list(s)], mode != "uniform", hr, y) for s in sets]
            res[str(t)][mode] = {**mean_metrics(rows), "n_subsets": len(sets)}
    return res


def load_members(files, iadrift=False):
    """{T: x [n, 256, 256, 3]}, {T: L [n]} for the files in order, up to the first missing one. With iadrift, also
    T = IAD from the members' _iad100 files, if every one of the n members has one."""
    xs, ls = {t: [] for t in STEPS}, {t: [] for t in STEPS}
    for p in files:
        if not p.exists():
            break
        _, a = load_npz(p)
        for t in STEPS:
            xs[t].append(a[f"x_hat_step{t}"])
            ls[t].append(float(a[f"lr_loss_step{t}"]))
    n = len(ls[STEPS[0]])
    iad_files = [p.with_name(p.stem + "_iad100.npz") for p in files[:n]]
    if iadrift and n and all(p.exists() for p in iad_files):
        xs[IAD], ls[IAD] = [], []
        for p in iad_files:
            _, a = load_npz(p)
            xs[IAD].append(a["x_step100"])
            ls[IAD].append(float(a["lr_loss_step100"]))
    return {t: np.asarray(v, np.float32) for t, v in xs.items()}, {t: np.asarray(v) for t, v in ls.items()}


def analyse_image(i, hr, y):
    a_files = [RUN_DIR / f"img{i:05d}_r{r:02d}.npz" for r in range(N_MEMBERS)]
    iad_files = [RUN_DIR / f"img{i:05d}_r{r:02d}_iad100.npz" for r in range(N_MEMBERS)]
    c_files = [RUN_DIR / f"img{i:05d}_r{r:02d}_shift{s[0]}{s[1]}.npz" for r in range(4) for s in SHIFTS]
    present = [p.name for p in a_files + iad_files + c_files if p.exists()]
    cache = CACHE_DIR / f"img{i:05d}.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached["files"] == present and cached.get("n_members_max") == N_MEMBERS:
            print(f"image {i}: cached analysis", flush=True)
            return cached
    xa, la = load_members(a_files, iadrift=True)
    n = len(la[STEPS[0]])
    budgets = [t for t in BUDGETS if t in la]
    out = {"image": i, "files": present, "n_members": n, "n_members_max": N_MEMBERS, "budgets": [str(t) for t in budgets],
           "bicubic": score(bicubic_up(y), hr, y)}
    if n == 0:
        return out
    finite = all(np.isfinite(xa[t]).all() for t in budgets)

    # single members, and the CSGM rule (the better of r0, r1 by LR loss; for IADrift, by the CSGM loss at 5,000)
    out["members"] = {str(t): [{"r": r, "lr_loss": float(la[t][r]), "psnr_raw": psnr(hr, xa[t][r])}
                               for r in range(n)] for t in budgets}
    out["csgm_rule"] = {}
    if n >= 2:
        for t in budgets:
            r = int(np.argmin(la[5_000 if t == IAD else t][:2]))
            out["csgm_rule"][str(t)] = {"kept": f"r{r}", **score(xa[t][r], hr, y)}

    out["k_curve"] = {}
    for k in KS:
        if k <= n:
            out["k_curve"][str(k)] = ensembles_over(subsets(n, k), xa, la, hr, y, budgets)
            print(f"image {i}: K = {k} done", flush=True)

    # arm C: 4 z x 4 grid phases (arm-A r0..3 unshifted, then the 12 shifted runs)
    xc, lc = load_members(c_files)
    if n >= 4 and len(lc[STEPS[0]]) == 12:
        grid_x = {t: np.concatenate([xa[t][:4], xc[t]]) for t in STEPS}
        grid_l = {t: np.concatenate([la[t][:4], lc[t]]) for t in STEPS}

        def cell(r, phase):  # phase 0 = unshifted, 1..3 = SHIFTS[phase - 1]
            return r if phase == 0 else 4 + 3 * r + phase - 1

        designs = {
            "z only (K=4: 4 z x 1 phase)": [[cell(r, 0) for r in range(4)]],
            "shifts only (K=4: 1 z x 4 phases)": [[cell(r, p) for p in range(4)] for r in range(4)],
            "both (K=8: 2 z x 4 phases)": [[cell(r, p) for r in rs for p in range(4)]
                                          for rs in itertools.combinations(range(4), 2)],
            "both (K=16: 4 z x 4 phases)": [[cell(r, p) for r in range(4) for p in range(4)]],
        }
        out["arm_c"] = {name: ensembles_over(sets, grid_x, grid_l, hr, y, ARM_C_STEPS) for name, sets in designs.items()}
        print(f"image {i}: arm C done", flush=True)
        finite = finite and all(np.isfinite(xc[t]).all() for t in STEPS)
    out["finite"] = bool(finite)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(cache, json.dumps(out).encode())
    return out


# ---------------------------------------------------------------- aggregation over images
def pick(m):
    keys = ("psnr_raw", "psnr", "lpips", "blockiness", "band_snr_db", "n_eff", "mmse_gain_db")
    return {k: m[k] for k in keys if k in m}


def avg(rows):
    return mean_metrics([pick(r) for r in rows])


def ref_metrics(abl, i, name):
    m = abl[str(i)][name]
    return {"psnr": m["psnr"], "lpips": m["lpips"], "blockiness": m["blockiness"], "band_snr_db": m["band_snr_db"]}


def aggregate(per_image, abl):
    full = [r for r in per_image if r.get("n_members") == N_MEMBERS]
    imgs = [r["image"] for r in full]
    agg = {"images": imgs, "n_images": len(imgs),
           "incomplete_images": {str(r["image"]): r.get("n_members", 0) for r in per_image if r not in full},
           "all_finite": all(r.get("finite", True) for r in per_image),
           "predicted_gain_db": {str(k): predicted_gain_db(k) for k in KS}}
    if not full:
        return agg
    budgets = [t for t in BUDGETS if all(str(t) in r["budgets"] for r in full)]
    agg["budgets"] = [str(t) for t in budgets]
    agg["iadrift_missing"] = [r["image"] for r in full if IAD not in r["budgets"]]
    agg["references"] = {name: {**{k: float(np.mean([ref_metrics(abl, i, name)[k] for i in imgs]))
                                   for k in ("psnr", "lpips", "blockiness")},
                                "band_snr_db": {b: float(np.mean([ref_metrics(abl, i, name)["band_snr_db"][b]
                                                                  for i in imgs])) for b in BANDS}}
                         for name in REFS}
    agg["references_note"] = ("All + BP. bicubic = BP(bicubic_up(y)), computed here; the rest from "
                              "results/ablation_metrics.json['ensembles'] (notebook 04, Section 5): CSGM = the better of r0, "
                              "r1 (5,000 steps); Ens-U / Ens-W = 5 x 5,000 steps (r0, r1 and 3 shifts of r0, so 2 distinct "
                              "z); IADrift @100 [Ens-U, Ens-W] = IADrift @100 from the same members (Ens-W weighted by the "
                              "CSGM losses)")
    agg["csgm_rule"] = {str(t): avg([r["csgm_rule"][str(t)] for r in full]) for t in budgets}
    agg["k_curve"] = {str(k): {str(t): {mode: avg([r["k_curve"][str(k)][str(t)][mode] for r in full])
                                        for mode in modes(t)} for t in budgets} for k in KS}

    # matched compute; per-image "better" counts use each image's subset-averaged metrics
    rows = []
    for name, k, t in MATCHED:
        if t not in budgets:
            continue
        for mode in modes(t):
            per = {r["image"]: r["k_curve"][str(k)][str(t)][mode] for r in full}
            row = {"ensemble": f"{name} {mode}", "K": k, "steps": str(t), "total_steps": total_steps(k, t),
                   **avg(list(per.values()))}
            for ref in REFS:
                row[f"better_psnr_than_{ref}"] = sum(per[i]["psnr"] > abl[str(i)][ref]["psnr"] for i in imgs)
                row[f"better_lpips_than_{ref}"] = sum(per[i]["lpips"] < abl[str(i)][ref]["lpips"] for i in imgs)
            rows.append(row)
    agg["matched_compute"] = rows

    with_c = [r for r in full if "arm_c" in r]
    if with_c:
        agg["arm_c_images"] = [r["image"] for r in with_c]
        arm_c = {name: {str(t): {mode: avg([r["arm_c"][name][str(t)][mode] for r in with_c])
                                 for mode in ("uniform", "weighted")} for t in ARM_C_STEPS}
                 for name in with_c[0]["arm_c"]}
        for k, label in ((8, "z only (K=8: 8 z x 1 phase, arm A subsets)"),
                         (12, "z only (K=12: 12 z x 1 phase, arm A)"),
                         (4, "z only (K=4: 4 z x 1 phase, arm A subsets)")):
            arm_c[label] = {str(t): {mode: avg([r["k_curve"][str(k)][str(t)][mode] for r in with_c])
                                     for mode in ("uniform", "weighted")} for t in ARM_C_STEPS}
        agg["arm_c"] = arm_c
    agg["per_image"] = {str(r["image"]): {
        "matched": {f"{name} {mode}": pick(r["k_curve"][str(k)][str(t)][mode])
                    for name, k, t in MATCHED if t in budgets for mode in modes(t)},
        "csgm_rule": {str(t): pick(r["csgm_rule"][str(t)]) for t in budgets},
        "references": {n: ref_metrics(abl, r["image"], n) for n in REFS}} for r in full}
    return agg


# ---------------------------------------------------------------- figures
COLORS = {500: "#2a78d6", 1_000: "#eb6834", 2_000: "#1baf7a", 5_000: "#eda100", IAD: "#e87ba4"}
BUDGET_LABEL = {500: "500 steps", 1_000: "1000 steps", 2_000: "2000 steps", 5_000: "5000 steps",
                IAD: "5000 + IADrift 100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="#fcfcfb")
    refs = (("bicubic", "bicubic + BP", "-"), ("CSGM", "CSGM + BP", ":"), ("CSGM Ens-U", "Ens-U (5 members) + BP", "-."),
            ("CSGM Ens-W", "Ens-W (5 members) + BP", "--"),
            ("IADrift @100 Ens-W", "IADrift @100 Ens-W + BP", (0, (1, 3))))
    budgets = [t for t in BUDGETS if str(t) in agg["budgets"]]
    for ax, key, title in ((axes[0], "psnr", "PSNR after BP (dB) ↑"), (axes[1], "lpips", "LPIPS after BP ↓")):
        ax.set_facecolor("#fcfcfb")
        for ref, _, ls in refs:
            ax.axhline(agg["references"][ref][key], color=MUTED, lw=1, ls=ls, zorder=1)
        for t in budgets:
            for mode, ls in (("uniform", "-"), ("weighted", "--")):
                ys = [agg["k_curve"][str(k)][str(t)][mode][key] for k in KS]
                ax.plot(KS, ys, ls, color=COLORS[t], lw=2, marker="o", ms=4, zorder=3,
                        label=BUDGET_LABEL[t] if mode == "uniform" else None)
        # reference labels at the right end of their lines, spaced so that close values stay legible
        lo, hi = ax.get_ylim()
        placed = []
        for ref, label, _ in sorted(refs, key=lambda r: agg["references"][r[0]][key]):
            v = agg["references"][ref][key]
            pos = (v - lo) / (hi - lo)
            if placed and pos - placed[-1] < 0.045:
                pos = placed[-1] + 0.045
            placed.append(pos)
            ax.annotate(label, xy=(1.0, (v - lo) / (hi - lo)), xycoords="axes fraction", xytext=(1.01, pos),
                        textcoords="axes fraction", va="center", fontsize=7.5, color=MUTED, annotation_clip=False)
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.set_xlabel("members K", color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=MUTED, ls="-"), plt.Line2D([], [], color=MUTED, ls="--")]
    labels += ["uniform mean", "weighted mean (LR loss)"]
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, fontsize=8)
    n = agg["n_images"]
    fig.suptitle(f"Ensembles of K members with distinct (ε, z), development images (n = {n}), not the test set",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 0.93, 0.95))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, facecolor=fig.get_facecolor())
    print(f"wrote {OUT_PNG}", flush=True)


# (name, K, T, weighted); the K members are r0..K-1 (fixed, never chosen with HR)
EXAMPLES = (("12 x 500 uniform", 12, 500, False), ("5 x 5000 uniform", 5, 5_000, False),
            ("12 x 5000 uniform", 12, 5_000, False), ("12 x (5000 + IADrift 100) uniform", 12, IAD, False))
CROP = (88, 88, 80)  # top, left, size of the zoomed centre crop (fixed, never chosen with HR)


def plot_examples(agg, index_rows, lr_all):
    """Per image: HR, bicubic + BP, CSGM + BP and the EXAMPLES ensembles + BP; full images and a zoomed crop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    imgs = agg["images"]
    examples = [e for e in EXAMPLES if str(e[2]) in agg["budgets"]]
    cols = ["HR", "bicubic + BP", "CSGM + BP"] + [f"{n} + BP" for n, _, _, _ in examples]
    fig, axes = plt.subplots(2 * len(imgs), len(cols), figsize=(2.3 * len(cols), 4.75 * len(imgs)),
                             facecolor="#fcfcfb", squeeze=False)
    top, left, size = CROP

    def show(x):
        return np.clip((np.asarray(x, np.float64) + 1) / 2, 0, 1)

    for row, i in enumerate(imgs):
        hr, y = data.hr_image(index_rows, i), np.asarray(lr_all[i], np.float32)
        xa, la = load_members([RUN_DIR / f"img{i:05d}_r{r:02d}.npz" for r in range(N_MEMBERS)], iadrift=True)
        r_kept = int(np.argmin(la[5_000][:2]))
        outs = [hr, backproject(bicubic_up(y), y), backproject(xa[5_000][r_kept], y)]
        for _, k, t, weighted in examples:
            outs.append(backproject(np.tensordot(weights(la[t][:k], weighted), xa[t][:k], axes=1), y))
        for c, (name, x) in enumerate(zip(cols, outs)):
            full, zoom = axes[2 * row, c], axes[2 * row + 1, c]
            full.imshow(show(x))
            zoom.imshow(show(x)[top:top + size, left:left + size], interpolation="nearest")
            full.add_patch(Rectangle((left, top), size, size, fill=False, ec="#eb6834", lw=1))
            if c == 0:
                title = f"{i}: {name}"
            else:
                title = f"{name.replace(' uniform', '')}\n{psnr(hr, x):.2f} dB, LPIPS {lpips_dist(x, hr):.3f}"
            full.set_title(title, fontsize=7.5, color=INK)
            for ax in (full, zoom):
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(GRID)
    fig.suptitle("Reconstructions vs bicubic, all + BP (development images, not the test set). Second row: zoomed "
                 "centre crop (orange box). Uniform means of members r0..rK-1", fontsize=9, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUT_EXAMPLES, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {OUT_EXAMPLES}", flush=True)


# ---------------------------------------------------------------- text tables
def print_tables(agg):
    def f(m):
        return (f"{m['psnr_raw']:6.2f} {m['psnr']:6.2f} {m['lpips']:6.3f} {m['blockiness']:5.2f} "
                f"{m['band_snr_db']['high']:6.2f} {m['band_snr_db']['very high']:6.2f} "
                f"{m.get('n_eff', float('nan')):5.1f}")

    head = "PSNRraw PSNRbp  LPIPS block  SNRhi SNRvhi  n_eff"
    print(f"\nImages {agg['images']} (n = {agg['n_images']}); incomplete {agg['incomplete_images']}; "
          f"all finite {agg['all_finite']}")
    print("\nReferences (+ BP; bicubic computed here, the rest from notebook 04, Section 5):  PSNR   LPIPS  block")
    for n, m in agg["references"].items():
        print(f"  {n:32s} {m['psnr']:6.2f} {m['lpips']:6.3f} {m['blockiness']:5.2f}")
    print(f"\nSingle member, CSGM rule (the better of r0, r1 by LR loss):  {head}")
    for t, m in agg["csgm_rule"].items():
        print(f"  T = {t:>6s}                                     {f(m)}")
    if agg["iadrift_missing"]:
        print(f"IADrift @100 missing for images {agg['iadrift_missing']}: budget iad100 left out")
    print(f"\nMatched compute:                              {head}  better PSNR/LPIPS vs Ens-W, vs CSGM, vs bicubic")
    for r in agg["matched_compute"]:
        print(f"  {r['ensemble']:42s} ({r['total_steps'] / 1000:4.1f}k) {f(r)}  "
              f"{r['better_psnr_than_CSGM Ens-W']}/{r['better_lpips_than_CSGM Ens-W']}, "
              f"{r['better_psnr_than_CSGM']}/{r['better_lpips_than_CSGM']}, "
              f"vs bicubic {r['better_psnr_than_bicubic']}/{r['better_lpips_than_bicubic']} of {agg['n_images']}")
    print("\nK-curve, PSNR after BP (uniform / weighted) and MMSE gain (uniform, before BP) vs prediction:")
    for t in [t for t in BUDGETS if str(t) in agg["budgets"]]:
        cells = [f"K{k}: {agg['k_curve'][str(k)][str(t)]['uniform']['psnr']:.2f}/"
                 f"{agg['k_curve'][str(k)][str(t)]['weighted']['psnr']:.2f}" for k in KS]
        print(f"  T = {t!s:>6s}  " + "  ".join(cells))
        cells = [f"K{k}: {agg['k_curve'][str(k)][str(t)]['uniform']['mmse_gain_db']:.2f}"
                 f"({predicted_gain_db(k):.2f})" for k in KS]
        print("     gain     " + "  ".join(cells))
    if "arm_c" in agg:
        print(f"\nArm C (images {agg['arm_c_images']}):                    {head}")
        for name, by_t in agg["arm_c"].items():
            for t, by_mode in by_t.items():
                for mode, m in by_mode.items():
                    print(f"  {name:44s} T={t:>4s} {mode[0]}  {f(m)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, nargs="+", default=IMAGES)
    args = parser.parse_args()
    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train, test = data.load_split(labels)
    assert set(args.images) <= set(IMAGES), "development images only"
    assert set(IMAGES) <= set(train.tolist()) and not set(IMAGES) & set(test.tolist()), "development images only"
    lr_all = data.load_measurements()
    abl = json.loads((paths.RESULTS_DIR / "ablation_metrics.json").read_text())["ensembles"]

    per_image = []
    for i in args.images:
        if not (RUN_DIR / f"img{i:05d}_r00.npz").exists():
            print(f"image {i}: no runs yet", flush=True)
            continue
        per_image.append(analyse_image(i, data.hr_image(index_rows, i), np.asarray(lr_all[i], np.float32)))
    for r in per_image:  # bicubic + BP is computed here and joins the reference rows
        abl[str(r["image"])]["bicubic"] = r["bicubic"]
    agg = aggregate(per_image, abl)
    agg["note"] = "Development only, post-hoc; not run on the test set."
    agg["config"] = {"budgets": [str(t) for t in BUDGETS], "ks": KS, "n_members": N_MEMBERS, "n_subsets": N_SUBSETS,
                     "subset_seed": SUBSET_SEED, "iadrift": "theta lr 1e-5, eps lr 1e-3, 100 steps from the step-5000 "
                                                         "CSGM solution (frozen IADrift @100)"}
    if agg["n_images"]:
        print_tables(agg)
        plot(agg)
        plot_examples(agg, index_rows, lr_all)
    atomic_write_bytes(OUT_JSON, (json.dumps(agg, indent=1) + "\n").encode())
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
