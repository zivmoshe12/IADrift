"""Print the tables and the numbers quoted in the text of the report, recomputed from results/ (CPU, seconds).

Usage:
    python scripts/report/report_numbers.py              # everything
    python scripts/report/report_numbers.py --only table1 averaging

Sections: table1 (test set), table2 (development diagnostics), table3 (late-stage ensembles, Appendix A),
text (LR loss, band SNR, error split, ensemble against bicubic), averaging (the averaging law, Sec. 4.4 and A.1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

from iadrift.runtime import setup_cpu

setup_cpu()

import argparse
import json

import numpy as np

from iadrift import paths
from iadrift.metrics import BANDS, RADIUS, WINDOW
from iadrift.stats import holm, paired_stats

R = paths.RESULTS_DIR


def load_json(name):
    return json.loads((R / name).read_text())


TEST = {  # row of Table 1 -> (file, key); every output is after BP
    "Bicubic": ("test", "bicubic + BP"),
    "CSGM": ("test", "CSGM, correct label + BP"),
    "IADrift@100": ("test", "IADrift @100, correct label + BP"),
    "IADrift@1000": ("test", "IADrift @1000, correct label + BP"),
    "Average of 2 restarts": ("ens", "E1: mean of 2 restarts + BP"),
    "Ens-W": ("ens", "Ens-W: weighted 5 CSGM runs + BP"),
    "Ens-U": ("ens", "Ens-U: mean of 5 CSGM runs + BP"),
}


class TestSet:
    def __init__(self):
        self.images = [s["image_index"] for s in load_json("test_sample.json")]
        self.files = {"test": load_json("test_metrics.json"), "ens": load_json("ensemble_test_metrics.json")}

    def metric(self, method, key, band=None):
        file, name = TEST[method]
        rows = [self.files[file][str(i)][name] for i in self.images]
        return np.array([r["band_snr_db"][band] if band else r[key] for r in rows])

    def info(self, i):
        return self.files["ens"][str(i)]["info"]


def fmt_stats(s, digits):
    f = f"{{:+.{digits}f}}"
    return f"{f.format(s['mean'])} [{f.format(s['lo'])}, {f.format(s['hi'])}] better on {int(s['better'])}"


def table1():
    t = TestSet()
    print("Table 1. Test set, 100 images, all after BP. Differences are paired, against CSGM.")
    stats = {}
    for method in TEST:
        line = f"  {method:22s} PSNR {t.metric(method, 'psnr').mean():6.2f}  LPIPS {t.metric(method, 'lpips').mean():.3f}"
        if method != "CSGM":
            stats[method] = {key: paired_stats(t.metric(method, key), t.metric("CSGM", key), key)
                             for key in ("psnr", "lpips")}
            line += (f"   dPSNR {fmt_stats(stats[method]['psnr'], 2)}   "
                     f"dLPIPS {fmt_stats(stats[method]['lpips'], 3)}")
        print(line)
    primary = [stats[m][k]["p"] for m in ("IADrift@1000", "IADrift@100") for k in ("psnr", "lpips")]
    print("  Holm over the 4 IADrift tests (@1000 PSNR, LPIPS, @100 PSNR, LPIPS):",
          ", ".join(f"{p:.1e}" for p in holm(primary)))
    print("  Holm over the 2 Ens-W tests (PSNR, LPIPS):",
          ", ".join(f"{p:.1e}" for p in holm([stats["Ens-W"][k]["p"] for k in ("psnr", "lpips")])))
    s = paired_stats(t.metric("Ens-U", "lpips"), t.metric("Ens-W", "lpips"), "lpips")
    print(f"  Ens-U - Ens-W: PSNR {t.metric('Ens-U', 'psnr').mean() - t.metric('Ens-W', 'psnr').mean():+.2f} dB, "
          f"LPIPS {fmt_stats(s, 3)}")


def table2():
    dev = load_json("dev_metrics.json")
    abl = load_json("ablation_metrics.json")
    traces = load_json("dev_traces.json")
    images = list(dev["images"])
    mean = lambda rows, key: np.mean([r[key] for r in rows])
    row = lambda name, rows: print(f"  {name:36s} {mean(rows, 'psnr'):6.2f}  " +
                                   (f"{mean(rows, 'lpips'):.3f}" if "lpips" in rows[0] else "  --"))
    print(f"Table 2. Development diagnostics, images {images}.")
    print("  4 images, SR")
    for name, key in (("Bicubic + BP", "bicubic + BP"), ("CSGM (no BP)", "CSGM"), ("CSGM + BP", "CSGM + BP")):
        row(name, [dev["images"][i][key] for i in images])
    row("  labels searched", [abl["noise_label_search"][i]["search"]["bp"] for i in images])
    row("  eps projected per patch", [abl["sphere"][i]["per-patch sphere"]["bp"] for i in images])
    row("  eps projected on sphere", [abl["sphere"][i]["global sphere"]["bp"] for i in images])
    row("IADrift@1000 (no BP)", [dev["images"][i]["IADrift"] for i in images])
    theta = [traces["theta_only"][f"{i}|1e-05"] for i in images]
    row("theta only, step 1,000 (no BP)", [{"psnr": t["psnr"][t["step"].index(1000)]} for t in theta])
    print("  same 4 images, HR fit (A = I)")
    row("CSGM-HR, frozen weights", [dev["images"][i]["CSGM-HR"] for i in images])
    row("IADrift-HR, adapted weights", [dev["images"][i]["IADrift-HR, lr 1e-05"] for i in images])
    print("  4 generated targets, SR")
    for name, key in (("Bicubic + BP", "bicubic + BP"), ("CSGM, true labels (no BP)", "CSGM, noise labels known"),
                      ("CSGM, random labels (no BP)", "CSGM, noise labels random")):
        row(name, [dev["r3"][i][key] for i in images])
    rms = [abl["sphere"][i]["none"]["eps"]["rms"] for i in images]
    print(f"  RMS of eps after CSGM: {min(rms):.1f} - {max(rms):.1f}")
    lr = [(abl["noise_label_search"][i]["baseline"]["raw"]["lr_loss"], abl["noise_label_search"][i]["search"]["raw"]["lr_loss"])
          for i in images]
    print(f"  label search, final LR loss: {np.mean([a for a, _ in lr]):.1f} -> {np.mean([b for _, b in lr]):.1f}")
    for lr_theta in ("1e-05", "1e-06"):
        gain = np.array([np.array(traces["theta_only"][f"{i}|{lr_theta}"]["psnr"]) for i in images])
        gain -= gain[:, :1]
        steps = traces["theta_only"][f"{images[0]}|{lr_theta}"]["step"]
        peak = int(np.argmax(gain.mean(0)))
        print(f"  theta only, lr {lr_theta}: mean PSNR gain peaks at {gain.mean(0)[peak]:+.2f} dB (step {steps[peak]}), "
              f"per-image peaks at steps {[steps[k] for k in gain.argmax(1)]}, "
              f"{gain.mean(0)[steps.index(1000)]:+.2f} dB at step 1,000")


def table3():
    z = load_json("z_ensemble_dev.json")
    refs = z["references"]
    print(f"Table 3. Late-stage exploration, development images {z['images']}, after BP; U / W.")
    print(f"  {'Bicubic':34s} {refs['bicubic']['psnr']:6.2f}          {refs['bicubic']['lpips']:.3f}")
    print(f"  {'CSGM':34s} {refs['CSGM']['psnr']:6.2f}          {refs['CSGM']['lpips']:.3f}")
    print(f"  {'IADrift@100':34s} {refs['IADrift @100']['psnr']:6.2f}          {refs['IADrift @100']['lpips']:.3f}")
    for name, u, w in (("Ens-U / Ens-W", "CSGM Ens-U", "CSGM Ens-W"),
                       ("same 5 members, each + IADrift@100", "IADrift @100 Ens-U", "IADrift @100 Ens-W")):
        print(f"  {name:34s} {refs[u]['psnr']:6.2f} / {refs[w]['psnr']:5.2f}  {refs[u]['lpips']:.3f} / {refs[w]['lpips']:.3f}")
    rows = {r["ensemble"]: r for r in z["matched_compute"]}
    for name in ("12 x 500", "12 x 1000", "12 x 2000", "5 x 5000", "12 x 5000", "5 x (5000 + IADrift 100)",
                 "12 x (5000 + IADrift 100)"):
        u, w = rows[f"{name} uniform"], rows[f"{name} weighted"]
        print(f"  {name:34s} {u['psnr']:6.2f} / {w['psnr']:5.2f}  {u['lpips']:.3f} / {w['lpips']:.3f}   "
              f"({u['total_steps'] / 1000:g}k steps)")
    best = rows["12 x (5000 + IADrift 100) weighted"]
    per = z["per_image"]
    lp = lambda name: np.array([per[str(i)]["matched"][name]["lpips"] for i in z["images"]])
    print(f"  12 x (5000 + IADrift@100) W lower LPIPS than 12 x 5000 W on "
          f"{int(np.sum(lp('12 x (5000 + IADrift 100) weighted') < lp('12 x 5000 weighted')))}/{len(z['images'])} images; "
          f"PSNR {best['psnr'] - rows['12 x 5000 weighted']['psnr']:+.2f} dB")


def text():
    t = TestSet()
    it = np.load(R / "iadrift_traces.npz")
    start, end = it["start_loss_oracle"], it["loss_oracle"][:, -1]
    print(f"LR loss during IADrift (test, step {int(it['steps'][-1]) + 1}): median {np.median(start):.1f} -> "
          f"{np.median(end):.2f}; more than tenfold on {int(np.sum(start / end > 10))}/{len(start)} images")
    print("Band SNR on the test set (dB), low / mid / high / very high:")
    for method in TEST:
        print(f"  {method:22s} " + " / ".join(f"{t.metric(method, None, b).mean():+5.1f}" for b, *_ in BANDS))
    gains = [t.metric("Ens-U", None, b).mean() - t.metric("CSGM", None, b).mean() for b, *_ in BANDS]
    print(f"  Ens-U - CSGM: {min(gains):+.1f} to {max(gains):+.1f} dB")

    # where the error that IADrift@1000 adds lies, on the four test images saved as examples: the increase of the
    # error energy over CSGM per band (Hann-windowed spectrum), summed over the images
    def band_energy(x):
        power = np.sum(np.abs(np.fft.fft2(x * WINDOW, axes=(0, 1))) ** 2, axis=-1)
        return np.array([power[(RADIUS >= lo) & (RADIUS < hi)].sum() for _, lo, hi in BANDS])

    with np.load(R / "examples_test.npz") as ex:
        energy = np.zeros(len(BANDS))
        for n in range(len(ex["image_index"])):
            hr = ex[f"{n}|hr"].astype(np.float64)
            error = lambda key: band_energy(ex[f"{n}|{key}, correct label + BP"].astype(np.float64) - hr)
            energy += error("IADrift @1000") - error("CSGM")
    share = energy / energy.sum()
    print(f"Error added by IADrift@1000 (4 example images): {100 * share[2:].sum():.1f}% above f_LR, "
          f"{100 * share[3]:.0f}% in the very-high band")

    for key in ("psnr", "lpips"):
        s = paired_stats(t.metric("Ens-U", key), t.metric("Bicubic", key), key)
        print(f"Ens-U - bicubic, {key}: {fmt_stats(s, 2 if key == 'psnr' else 3)}")


def rho_of(gain, k):
    return (2 / 10 ** (gain / 10) - 1 - 1 / k) * k / (k - 1)


def beta_of(gain, k):
    r = 10 ** (gain / 10)
    return (r - 1) / (1 - r / k)


def averaging():
    t = TestSet()
    to_db = lambda x: 10 * np.log10(x)
    per_image, members, ensemble, restarts, shifts = [], [], [], [], []
    for i in t.images:
        info = t.info(i)
        m, e = info["member_mse"], info["Ens-U: mean of 5 CSGM runs"]["mse_raw"]
        per_image.append(to_db(np.mean(list(m.values())) / e))
        members.append(np.mean(list(m.values())))
        ensemble.append(e)
        restarts.append((m["r0"] + m["r1"]) / 2)
        shifts.append(np.mean([m["shift01"], m["shift10"], m["shift11"]]))
    g = np.array(per_image)
    half = 1.96 * g.std(ddof=1) / np.sqrt(len(g))
    restarts, shifts, ensemble = map(np.array, (restarts, shifts, ensemble))
    print(f"Ens-U gain over its mean member, test set, before BP: {g.mean():.2f} dB [{g.mean() - half:.2f}, "
          f"{g.mean() + half:.2f}] per image; {to_db(np.mean(members) / np.mean(ensemble)):.2f} dB pooled; "
          f"predicted {to_db(2 / (1 + 1 / 5)):.2f} dB")
    print(f"  shifted members worse than the restarts by {np.mean(to_db(shifts / restarts)):.2f} dB, on "
          f"{int(np.sum(shifts > restarts))}/{len(g)} images; gain against the restarts alone "
          f"{np.mean(to_db(restarts / ensemble)):.2f} dB")

    z = load_json("z_ensemble_dev.json")
    ks = [int(k) for k in z["k_curve"] if int(k) > 1]
    print("Late-stage experiment, gain before BP (uniform) against the prediction, by member length:")
    for steps in ("500", "1000", "2000", "5000"):
        gains = [z["k_curve"][str(k)][steps]["uniform"]["mmse_gain_db"] for k in ks]
        line = "  ".join(f"K{k}: {gk:.2f} ({z['predicted_gain_db'][str(k)]:.2f})" for k, gk in zip(ks, gains))
        print(f"  {steps:>5s} steps  {line}")
        rho = [rho_of(gk, k) for k, gk in zip(ks, gains)]
        beta = [beta_of(gk, k) for k, gk in zip(ks, gains)]
        print(f"               rho {min(rho):.2f} - {max(rho):.2f}, beta {min(beta):.2f} - {max(beta):.2f}")
    per = [z["per_image"][str(i)] for i in z["images"]]
    if "k_curve" in per[0]:
        print("  per-image K=5 gains, 5,000 steps:",
              ", ".join(f"{p['k_curve']['5']['5000']['uniform']['mmse_gain_db']:.2f}" for p in per))


SECTIONS = {"table1": table1, "table2": table2, "table3": table3, "text": text, "averaging": averaging}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=list(SECTIONS))
    args = parser.parse_args()
    for name in args.only or SECTIONS:
        SECTIONS[name]()
        print()


if __name__ == "__main__":
    main()
