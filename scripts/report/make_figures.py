"""Draw the figures of the report and the full report from results/ (CPU, under a minute).

Usage:
    python scripts/report/make_figures.py                 # all figures -> figures/report/*.pdf
    python scripts/report/make_figures.py --only band_snr_test examples_test
    python scripts/report/make_figures.py --png           # also write a PNG next to every PDF

Figures (report = the 7-page report, full = the full report):
    examples_test        report Fig. 1, full Fig. 4   one test image, every method of Table 1, after BP
    band_snr_test        report Fig. 2, full Fig. 3   band SNR on the test set, mean and 95% CI
    paired_test          full Fig. 2                  per-image differences from CSGM + BP
    fit_y_vs_x           full Fig. 5                  LR loss and PSNR during adaptation, SR and HR fit
    representation       full Fig. 6                  HR fit and in-range targets (development, 4 images)
    averaging_law        report Fig. 3, full Fig. 7   averaging gain and LPIPS against K (development, 6 images)
    z_iad_examples       report Fig. 4                12-member averages, crops (needs results/z_ensemble_examples.npz)
    z_ensemble_examples  full Fig. 8                  the same, full images and crops

The method diagram of the full report (Fig. 1) is drawn by hand: figures/methods_overview.pdf.

Fonts: DejaVu Serif, which ships with matplotlib, so the figures look the same on every machine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

from iadrift.runtime import setup_cpu

setup_cpu()

import argparse
import json

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

from iadrift import paths
from iadrift.operators import backproject
from iadrift.stats import mean_ci

OUT_DIR = paths.FIGURES_DIR / "report"
R = paths.RESULTS_DIR

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

COLOR = {"bicubic": "#8c8c8c", "CSGM": "#3a73b0", "IADrift@100": "#e9a756", "IADrift@1000": "#b6432f",
         "2 restarts": "#8f68b8", "Ens-W": "#5bbccd", "Ens-U": "#4f9d3f"}

# display name -> (file, key) for the test-set methods; every key is the "+ BP" output
TEST_KEYS = {
    "bicubic": ("test", "bicubic + BP"),
    "CSGM": ("test", "CSGM, correct label + BP"),
    "IADrift@100": ("test", "IADrift @100, correct label + BP"),
    "IADrift@1000": ("test", "IADrift @1000, correct label + BP"),
    "2 restarts": ("ens", "E1: mean of 2 restarts + BP"),
    "Ens-W": ("ens", "Ens-W: weighted 5 CSGM runs + BP"),
    "Ens-U": ("ens", "Ens-U: mean of 5 CSGM runs + BP"),
}
BANDS = (("low", "low [0, 1/16)\n(below $f_{LR}$)"), ("mid", "mid [1/16, 1/8)\n(below $f_{LR}$)"),
         ("high", "high [1/8, 1/4)\n(above $f_{LR}$)"), ("very high", "very high $\\geq 1/4$\n(above $f_{LR}$)"))
CROP = 80  # zoomed centre crop of the development examples


def load_json(name):
    return json.loads((R / name).read_text())


def load_npz(name):
    with np.load(R / name) as f:
        return {k: f[k] for k in f.files}


class TestSet:
    def __init__(self):
        self.images = [s["image_index"] for s in load_json("test_sample.json")]
        self.files = {"test": load_json("test_metrics.json"), "ens": load_json("ensemble_test_metrics.json")}

    def metric(self, method, key, band=None):
        file, name = TEST_KEYS[method]
        rows = [self.files[file][str(i)][name] for i in self.images]
        return np.array([r["band_snr_db"][band] if band else r[key] for r in rows])


def show(x):
    return np.clip((np.asarray(x, np.float32) + 1) / 2, 0, 1)


def save(fig, name, png):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf")
    if png:
        fig.savefig(OUT_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT_DIR / name}.pdf")


# ---------------------------------------------------------------------------------------------------- test set
def band_snr_test(png):
    t = TestSet()
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 1.75))
    for ax, (band, title) in zip(axes, BANDS):
        for k, method in enumerate(TEST_KEYS):
            m, half = mean_ci(t.metric(method, None, band))
            ax.errorbar(k, m, yerr=half, fmt="o", ms=5, capsize=2, color=COLOR[method], label=method)
        if band in ("high", "very high"):
            ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(TEST_KEYS) - 0.4)
    axes[0].set_ylabel("band SNR (dB)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[m], label=m) for m in TEST_KEYS]
    fig.legend(handles=handles, loc="lower center", ncol=len(TEST_KEYS), frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(w_pad=1.5)
    save(fig, "band_snr_test", png)


def paired_test(png):
    t = TestSet()
    shown = [("bicubic", "bicubic\n+BP", "secondary"), ("IADrift@100", "IADrift\n@100", "primary"),
             ("IADrift@1000", "IADrift\n@1000", "primary"), ("Ens-W", "Ens-W", "primary"),
             ("Ens-U", "Ens-U", "secondary")]
    rng = np.random.default_rng(0)  # horizontal jitter of the dots only
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9))
    for ax, key, ylabel in ((axes[0], "psnr", "PSNR $-$ CSGM+BP (dB) $\\uparrow$"),
                            (axes[1], "lpips", "LPIPS $-$ CSGM+BP $\\downarrow$")):
        ref = t.metric("CSGM", key)
        for k, (method, _, status) in enumerate(shown):
            d = t.metric(method, key) - ref
            half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))  # the interval of iadrift.stats.paired_stats
            ax.scatter(k + rng.uniform(-0.25, 0.25, len(d)), d, s=3, alpha=0.5, color=COLOR[method], lw=0)
            ax.errorbar(k + 0.4, d.mean(), yerr=half, fmt="o", ms=4, capsize=2, color="k")
            better = int(np.sum(d > 0 if key == "psnr" else d < 0))
            ax.text(k, 1.02, f"{better}/{len(d)}\n{status}", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=6.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(shown)), [label for _, label, _ in shown])
        ax.set_ylabel(ylabel)
    fig.text(0.5, -0.03, "dots: one test image each; black: mean with 95% CI of the paired difference; "
             "above: images better than CSGM+BP", ha="center", fontsize=6.5)
    fig.tight_layout(w_pad=2)
    save(fig, "paired_test", png)


def examples_test(png, n=1):
    """Test example n of results/examples_test.npz (n = 1: the first was used only to check the code)."""
    t = TestSet()
    ex, ens = load_npz("examples_test.npz"), load_npz("ensemble_examples.npz")
    i = int(ex["image_index"][n])
    assert int(ens["image_index"][n]) == i
    get = lambda f, key: f[f"{n}|{key}"].astype(np.float32)
    y = get(ex, "lr")
    two_restarts = backproject((get(ens, "r0") + get(ens, "r1")) / 2, y)  # stored as members, so BP is redone here
    k = t.images.index(i)
    caption = lambda method: (f"{t.metric(method, 'psnr')[k]:.2f} dB / {t.metric(method, 'lpips')[k]:.3f}")
    panels = [
        [("HR", get(ex, "hr"), f"test {i}"),
         ("bicubic + BP", get(ex, "bicubic + BP"), caption("bicubic")),
         ("CSGM + BP", get(ex, "CSGM, correct label + BP"), caption("CSGM")),
         ("IADrift@100 + BP", get(ex, "IADrift @100, correct label + BP"), caption("IADrift@100")),
         ("IADrift@1000 + BP", get(ex, "IADrift @1000, correct label + BP"), caption("IADrift@1000"))],
        [("input $y$ (64$\\times$64)", y, ""),
         ("average of 2 restarts", two_restarts, caption("2 restarts")),
         ("Ens-W", get(ens, "Ens-W + BP"), caption("Ens-W")),
         ("Ens-U", get(ens, "Ens-U + BP"), caption("Ens-U")),
         None]]
    r0, c0, size = 100, 92, 64  # the yellow box, in HR pixels
    fig, axes = plt.subplots(2, 5, figsize=(8.0, 4.0))
    for row, ax_row in zip(panels, axes):
        for panel, ax in zip(row, ax_row):
            ax.axis("off")
            if panel is None:
                continue
            title, x, text = panel
            big = np.repeat(np.repeat(x, 4, 0), 4, 1) if x.shape[0] == 64 else x
            ax.imshow(show(big), interpolation="nearest")
            ax.add_patch(plt.Rectangle((c0, r0), size, size, fill=False, color="#ffff55", lw=1))
            inset = ax.inset_axes([0.58, 0.0, 0.42, 0.42])
            inset.imshow(show(big[r0:r0 + size, c0:c0 + size]), interpolation="nearest")
            inset.set_xticks([]), inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color("#ffff55")
                spine.set_linewidth(1.5)
            ax.set_title(title, fontsize=8, pad=2)
            ax.text(0.5, -0.03, text, transform=ax.transAxes, ha="center", va="top", fontsize=7)
    fig.tight_layout(h_pad=1.2, w_pad=0.3)
    save(fig, "examples_test", png)


# ---------------------------------------------------------------------------------------------- development
def fit_y_vs_x(png):
    traces = load_json("dev_traces.json")
    it = load_npz("iadrift_traces.npz")
    dev = load_json("dev_metrics.json")
    images = list(dev["images"])
    theta = [traces["theta_only"][f"{i}|1e-05"] for i in images]
    hr_fit = [traces["hr_fit"][f"{i}|1e-05"] for i in images]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 4.0), sharex="col")
    steps = np.asarray(theta[0]["step"])
    loss = np.array([t["lr_loss"] for t in theta])
    for row in loss:
        axes[0, 0].plot(steps, row, color="0.75", lw=0.7)
    axes[0, 0].plot(steps, np.exp(np.log(loss).mean(0)), color="#7b2f9e", lw=1.6,
                    label=f"dev, geometric mean (n={len(images)})")
    test_steps = np.concatenate([[0], it["steps"] + 1])
    test_loss = np.concatenate([it["start_loss_oracle"][:, None], it["loss_oracle"]], axis=1)
    axes[0, 0].plot(test_steps, np.median(test_loss, 0), "k--", lw=1,
                    label=f"test, joint IADrift, median (n={len(test_loss)})")
    axes[0, 0].set(yscale="log", ylabel="LR loss $\\|y - Ax\\|^2$", title="(a) SR: fit $y$ ($\\theta$-only, lr $10^{-5}$)")
    axes[0, 0].legend(frameon=False)
    psnr = np.array([t["psnr"] for t in theta])
    for row in psnr:
        axes[1, 0].plot(steps, row, color="0.75", lw=0.7)
    axes[1, 0].plot(steps, psnr.mean(0), color="#b6432f", lw=1.6)
    axes[1, 0].axhline(np.mean([dev["images"][str(i)]["CSGM"]["psnr"] for i in images]), color="k", ls=":", lw=0.6)
    axes[1, 0].set(xlabel="adaptation step", ylabel="PSNR, no BP (dB)")

    steps = np.asarray(hr_fit[0]["step"])
    loss = np.array([t["hr_loss"] for t in hr_fit])
    for row in loss:
        axes[0, 1].plot(steps, row, color="0.75", lw=0.7)
    axes[0, 1].plot(steps, np.exp(np.log(loss).mean(0)), color="#7b2f9e", lw=1.6,
                    label=f"dev, geometric mean (n={len(images)})")
    axes[0, 1].set(yscale="log", ylabel="HR loss $\\|x^\\star - x\\|^2$",
                   title="(b) fit the HR image ($A = I$, joint, lr $10^{-5}$)")
    axes[0, 1].legend(frameon=False)
    psnr = np.array([t["psnr"] for t in hr_fit])
    for row in psnr:
        axes[1, 1].plot(steps, row, color="0.75", lw=0.7)
    axes[1, 1].plot(steps, psnr.mean(0), color="#4f9d3f", lw=1.6)
    axes[1, 1].set(xlabel="adaptation step", ylabel="PSNR, no BP (dB)")
    fig.tight_layout()
    save(fig, "fit_y_vs_x", png)


def representation(png):
    dev = load_json("dev_metrics.json")
    images = list(dev["images"])
    real = [("bicubic\n+BP", "bicubic + BP", "#8c8c8c"), ("CSGM\n+BP", "CSGM + BP", "#3a73b0"),
            ("CSGM\n-HR", "CSGM-HR", "#7aaad3"), ("IADrift\n-HR", "IADrift-HR, lr 1e-05", "#4f9d3f")]
    in_range = [("bicubic\n+BP", "bicubic + BP", "#8c8c8c"), ("CSGM\n$s$ known", "CSGM, noise labels known", "#8f68b8"),
                ("CSGM\n$s$ random", "CSGM, noise labels random", "#3a73b0")]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.4))
    for ax, key, ylabel in ((axes[0], "psnr", "PSNR (dB) $\\uparrow$"), (axes[1], "lpips", "LPIPS $\\downarrow$")):
        x, ticks = 0, []
        for group, source in ((real, dev["images"]), (in_range, dev["r3"])):
            for label, name, color in group:
                v = np.array([source[str(i)][name][key] for i in images])
                ax.bar(x, v.mean(), 0.75, color=color)
                ax.scatter(np.full(len(v), x), v, s=6, color="k", zorder=3)
                ticks.append((x, label))
                x += 1
            x += 0.5
        ax.axvline(len(real) - 0.25, color="k", ls=":", lw=0.6)
        ax.text((len(real) - 1) / 2, 1.0, "real photos", transform=ax.get_xaxis_transform(), ha="center", va="bottom")
        ax.text(len(real) + 1.5, 1.0, "in-range target $x^\\star = G(\\varepsilon^\\star, s^\\star)$",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom")
        ax.set_xticks([t for t, _ in ticks], [lab for _, lab in ticks], fontsize=6)
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(15, None)
    fig.tight_layout(w_pad=2)
    save(fig, "representation", png)


def averaging_law(png):
    z = load_json("z_ensemble_dev.json")
    ks = [int(k) for k in z["k_curve"]]
    curve = lambda t, mode, key: [z["k_curve"][str(k)][t][mode][key] for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.6), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]
    ax.plot(ks, [z["predicted_gain_db"][str(k)] for k in ks], "k--", lw=1.2,
            label="prediction $10\\log_{10}\\frac{2}{1+1/K}$")
    for t, color in (("500", "#efb26a"), ("1000", "#e1764f"), ("2000", "#83aed2"), ("5000", "#2e3586")):
        ax.plot(ks, curve(t, "uniform", "mmse_gain_db"), "o-", ms=3, lw=1.6, color=color,
                label=f"measured, {int(t):,}-step members")
    ax.set(xticks=ks, xlabel="ensemble size $K$", ylabel="gain, before BP (dB)",
           title="(a) averaging gain over the mean member")
    ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    ax.plot(ks, curve("500", "uniform", "lpips"), "o-", ms=3, color="#efb26a", label="500-step members, U")
    ax.plot(ks, curve("5000", "uniform", "lpips"), "o-", ms=3, color="#2e3586", label="5,000-step members, U")
    ax.plot(ks, curve("5000", "weighted", "lpips"), "o--", ms=3, color="#2e3586", label="5,000-step members, W")
    ax.plot(ks, curve("iad100", "uniform", "lpips"), "o-", ms=3, color="#6a2a9a", label="5,000 + IADrift@100, U")
    ax.plot(ks, curve("iad100", "weighted", "lpips"), "o--", ms=3, color="#6a2a9a", label="5,000 + IADrift@100, W")
    refs = z["references"]
    ax.plot(5, refs["CSGM Ens-U"]["lpips"], "s", ms=6, color=COLOR["Ens-U"], mec="k", mew=0.6,
            label="Ens-U: 2 starts + 3 shifts")
    ax.plot(5, refs["CSGM Ens-W"]["lpips"], "D", ms=6, color=COLOR["Ens-W"], mec="k", mew=0.6,
            label="Ens-W: 2 starts + 3 shifts")
    ax.axhline(refs["CSGM"]["lpips"], color=COLOR["CSGM"], ls=":", lw=1)
    ax.text(12.15, refs["CSGM"]["lpips"], "CSGM+BP", color=COLOR["CSGM"], va="center", fontsize=6.5)
    ax.axvline(5, color="0.7", lw=0.6)
    ax.text(5.15, 0.98, "$K=5$: 25k steps,\nthe cost of Ens-U/W", transform=ax.get_xaxis_transform(), va="top",
            fontsize=6.5, color="0.4")
    ax.set(xticks=ks, xlim=(0.5, 14.5), xlabel="ensemble size $K$", ylabel="LPIPS after BP $\\downarrow$",
           title="(b) LPIPS, members with distinct labels $s$")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5)
    fig.tight_layout()
    save(fig, "averaging_law", png)


def z_examples(png):
    path = R / "z_ensemble_examples.npz"
    if not path.exists():
        print(f"skipped z_iad_examples and z_ensemble_examples: {path} is missing; it is written by "
              "scripts/analysis/export_z_ensemble_results.py, which needs the development caches")
        return
    ex = load_npz("z_ensemble_examples.npz")
    images = [int(i) for i in ex["image_index"]]
    a = slice(128 - CROP // 2, 128 + CROP // 2)  # centre crop
    psnr_lpips = lambda i, key: f"{ex[f'{i}|{key}|metrics'][0]:.2f} dB / {ex[f'{i}|{key}|metrics'][1]:.3f}"

    cols = [("HR", "hr"), ("bicubic + BP", "bicubic + BP"), ("CSGM + BP\n(10k steps)", "CSGM + BP"),
            ("12 $\\times$ 500, U\n(6k steps)", "12 x 500"), ("12 $\\times$ 5k, U\n(60k steps)", "12 x 5000"),
            ("12 $\\times$ (5k + IADrift@100), U\n(61.2k steps)", "12 x (5000 + IADrift 100)")]
    fig, axes = plt.subplots(len(images), len(cols), figsize=(8.0, 1.5 * len(images) + 0.4))
    for r, i in enumerate(images):
        for c, (title, key) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(show(ex[f"{i}|{key}"][a, a]), interpolation="nearest")
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=7.5)
            ax.text(0.5, -0.02, f"dev {i}" if key == "hr" else psnr_lpips(i, key), transform=ax.transAxes,
                    ha="center", va="top", fontsize=7)
    fig.tight_layout(h_pad=0.9, w_pad=0.2)
    save(fig, "z_iad_examples", png)

    cols = [("HR", "hr"), ("bicubic + BP", "bicubic + BP"), ("CSGM + BP\n10k steps", "CSGM + BP"),
            ("5 runs\n25k steps", "5 x 5000"), ("12 runs\n60k steps", "12 x 5000"),
            ("12 runs + IADrift@100\n61.2k steps", "12 x (5000 + IADrift 100)")]
    fig, axes = plt.subplots(2 * len(images), len(cols), figsize=(8.0, 2.75 * len(images) + 0.5))
    for r, i in enumerate(images):
        for c, (title, key) in enumerate(cols):
            x = ex[f"{i}|{key}"]
            full, zoom = axes[2 * r, c], axes[2 * r + 1, c]
            full.imshow(show(x))
            full.add_patch(plt.Rectangle((a.start, a.start), CROP, CROP, fill=False, color="#e07b4f", lw=0.8))
            zoom.imshow(show(x[a, a]), interpolation="nearest")
            full.axis("off"), zoom.axis("off")
            if r == 0:
                full.set_title(title, fontsize=7.5)
            full.text(0.5, -0.02, f"dev {i}" if key == "hr" else psnr_lpips(i, key), transform=full.transAxes,
                      ha="center", va="top", fontsize=7)
    fig.text(0.75, 1.0, "average of 5,000-step CSGM runs, each with its own $(\\varepsilon, s)$", ha="center",
             va="bottom", fontsize=7.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.2)
    save(fig, "z_ensemble_examples", png)


FIGURES = {"examples_test": examples_test, "band_snr_test": band_snr_test, "paired_test": paired_test,
           "fit_y_vs_x": fit_y_vs_x, "representation": representation, "averaging_law": averaging_law,
           "z_examples": z_examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=list(FIGURES), help="draw only these figures")
    parser.add_argument("--png", action="store_true", help="also write PNG files")
    args = parser.parse_args()
    for name in args.only or FIGURES:
        FIGURES[name](args.png)


if __name__ == "__main__":
    main()
