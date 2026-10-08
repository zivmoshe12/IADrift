"""Noisy SR x4 (sigma = 10/255): CSGM with lambda ||eps||^2 and regularized back-projection (development, 6 images).

Development-only result (not in the notebooks). Images: the 6 development images of the ensemble checks (8117, 3939,
403, 347, 9780, 8405), correct label. Measurement y = A HR + sigma n with one fixed noise draw per image
(iadrift.noisy: sigma = 2 * 10/255 in [-1, 1] units, n from fold_in(PRNGKey(2030), image), stored once in
imageNet_x4_noisy_s10.npz together with the 100 test images).

    CSGM:  min_eps ||y - A G(eps)||^2 + lambda ||eps||^2,  lambda in {0, 0.1 sigma^2, sigma^2, 10 sigma^2}
           (MAP weight: sigma^2), Adam lr 0.02, 5,000 steps, restarts r0, r1 from initialize(i, r) -> 48 runs.
           Per lambda and image the restart with the lowest final DATA term ||y - A G(eps)||^2 is kept (never HR).
    BP:    x + A^T (A A^T + eta I)^{-1} (y - A x), eta in {0, 0.01, 0.03, 0.1, 0.3} x lambda_max(A A^T), applied to
           every output incl. bicubic.

Selection rule (stated before looking at any result):
    eta*    = argmax over eta of the mean development PSNR of bicubic + BP_eta (prior-free);
    lambda* = argmax over lambda of the mean development PSNR of CSGM(lambda) + BP_eta*,
              ties (equal mean PSNR at 0.01 dB) broken by the lower mean LPIPS.

Ported from scripts/run_sr4_noisy_dev.py and scripts/noisy_sr4.py (now iadrift.noisy; NoisyCSGM.reconstruct is the
original reconstruct_noisy, and lambda = 0 runs the frozen CSGM step itself); outputs are the cache the published
results were computed from. Result (grid.json): eta* = 0.3 lambda_max, lambda* = sigma^2.

Usage:
    python scripts/dev/run_noisy_dev.py --dry-run            # plan and cache status; runs and writes nothing (CPU)
    python scripts/dev/run_noisy_dev.py --smoke              # GPU: 100-step trace prefixes + lambda-0 = frozen CSGM
    python scripts/dev/run_noisy_dev.py --max-runs 10        # GPU sweep, resumable (one .npz per run)
    python scripts/dev/run_noisy_dev.py --evaluate           # BP grid, metrics, LPIPS, selection -> grid.json
    python scripts/dev/run_noisy_dev.py --evaluate --check   # the same on the CPU, compared with grid.json; no writes

The GPU sweep prints ALL RUNS COMPLETE when all 48 runs are saved.
Outputs (IADRIFT_DATA/csgm_noisy_dev_v1/): img#####_lam<tag>_r#.npz (metrics_json {lam, initial_data, final_data,
final_penalty, final_eps_rms, noise_floor, psnr, seconds}; eps, noise_labels, x_hat, trace_steps, trace_loss,
trace_data, psnr_steps, psnr_trace), grid.json (selection, means and per-image metrics of every output and BP variant).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

CHECK = "--check" in sys.argv
CPU = "--dry-run" in sys.argv or CHECK
if CPU:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from iadrift.runtime import setup

setup(theta_gradients=False, require_gpu=not CPU)  # CSGM only: default allocator

import argparse
import json

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import atomic_write_bytes, load_npz, save_npz
from iadrift.csgm import FROZEN_CONFIG

SIGMA = 2 * 10 / 255  # = iadrift.noisy.SIGMA (not imported here: that module imports JAX)
OUT_DIR = paths.DATA_DIR / "csgm_noisy_dev_v1"
IMAGES = list(data.DEV_IMAGES_6)
NUM_RESTARTS = FROZEN_CONFIG["num_restarts"]
LAMBDAS = {"0": 0.0, "0.1s2": 0.1 * SIGMA ** 2, "1s2": SIGMA ** 2, "10s2": 10 * SIGMA ** 2}
ETA_FACTORS = (0.0, 0.01, 0.03, 0.1, 0.3)
SMOKE = (8117, 0, 100)  # image, restart, steps
SELECTION_RULE = (  # the text stored in grid.json (from the original docstring)
    "(stated before looking at any result):\n"
    "    eta*    = argmax over eta of the mean development PSNR of bicubic + BP_eta (prior-free);\n"
    "    lambda* = argmax over lambda of the mean development PSNR of CSGM(lambda) + BP_eta*,\n"
    "              ties (equal mean PSNR at 0.01 dB) broken by the lower mean LPIPS.\n"
    "Per lambda and image, the restart with the lowest final DATA term ||y - A f(eps)||^2 is kept (never HR).")


def run_path(i, tag, r):
    return OUT_DIR / f"img{i:05d}_lam{tag}_r{r}.npz"


def plan():
    return [(i, tag, r) for i in IMAGES for tag in LAMBDAS for r in range(NUM_RESTARTS)]


def measurement_indices(labels):
    """The images with a stored noisy y: the 6 development images + the 100-image test sample."""
    _, test_indices = data.load_split(labels)
    return IMAGES + [int(i) for i in data.get_test_sample(labels, test_indices)]


def load_noisy_y(labels, lr_all):
    """GPU: iadrift.noisy.noisy_measurements (stored y, checked against a fresh draw). On the CPU the fresh normal draw
    can differ from the GPU's in the last bit, so --check reads the stored values and reports that difference."""
    from iadrift import noisy

    if not CHECK:
        return noisy.noisy_measurements(measurement_indices(labels), lr_all)
    with np.load(paths.DATA_DIR / noisy.NOISY_Y_NPZ) as d:
        stored = dict(zip(d["indices"].tolist(), d["y_noisy"]))
    diff = max(float(np.max(np.abs(np.asarray(lr_all[i], np.float32) + np.float32(SIGMA) * noisy.noise_draw(i)
                                   - stored[i]))) for i in IMAGES)
    print(f"stored noisy y vs a fresh CPU draw: max |difference| {diff:.1e} (the stored values are used)")
    return {i: stored[i] for i in IMAGES}


def sweep(args, index_rows, labels, y_noisy, ncsgm):
    todo = [p for p in plan() if not run_path(*p).exists()]
    print(f"Saved {len(plan()) - len(todo)} / {len(plan())}; this call: {min(len(todo), args.max_runs)}")
    from iadrift.noisy import NOISE_FLOOR
    for i, tag, r in todo[: args.max_runs]:
        hr = data.hr_image(index_rows, i)
        m, a = ncsgm.reconstruct(i, r, int(labels[i]), y_noisy[i], hr, LAMBDAS[tag])
        if not (np.isfinite(m["final_data"]) and np.all(np.isfinite(a["x_hat"]))):
            raise RuntimeError(f"Non-finite result: image {i} lambda {tag} r{r}; rerun identically.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        save_npz(run_path(i, tag, r), a, m)
        print(f"image {i} lam {tag:>5} r{r}: data {m['final_data']:.1f} (floor {NOISE_FLOOR:.1f}), "
              f"penalty {m['final_penalty']:.1f}, eps rms {m['final_eps_rms']:.3f}, PSNR {m['psnr']:.2f}, "
              f"{m['seconds']:.0f}s", flush=True)
    if all(run_path(*p).exists() for p in plan()):
        print("ALL RUNS COMPLETE")


def smoke(index_rows, labels, y_noisy, ncsgm):
    """100 steps from image 8117, restart 0: trace prefixes of the saved lambda = 0 and sigma^2 runs, and lambda = 0
    against the frozen CSGM loop. Writes nothing."""
    i, r, steps = SMOKE
    hr = data.hr_image(index_rows, i)
    ok = True
    for tag in ("0", "1s2"):
        m, a = ncsgm.reconstruct(i, r, int(labels[i]), y_noisy[i], hr, LAMBDAS[tag], num_steps=steps)
        _, saved = load_npz(run_path(i, tag, r))
        n = len(a["trace_loss"])
        same = all(np.array_equal(a[k], saved[k][:n]) for k in ("trace_loss", "trace_data"))
        print(f"{run_path(i, tag, r).name}, first {steps} steps: trace {a['trace_loss'].tolist()} vs saved "
              f"{saved['trace_loss'][:n].tolist()}; identical: {same}")
        ok &= same
        if tag == "0":
            mb, ab = ncsgm.csgm.reconstruct(i, r, int(labels[i]), y_noisy[i], hr, num_steps=steps)
            frozen = m["final_data"] == mb["final_loss"] and np.array_equal(a["eps"], ab["eps"]) and \
                np.array_equal(a["x_hat"], ab["x_hat"])
            print(f"lambda 0 vs frozen CSGM.reconstruct ({steps} steps): identical: {frozen}")
            ok &= frozen
    print("SMOKE PASSED" if ok else "SMOKE FAILED")
    raise SystemExit(0 if ok else 1)


# ---------------------------------------------------------------- evaluation
def make_backproject():
    """The original noisy_sr4.backproject: one jitted function with eta as an argument for every eta, including 0
    (iadrift.operators.backproject compiles a separate function for eta = 0, so it is not used here)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg
    from iadrift.operators import A_sr4, A_transpose

    @jax.jit
    def _backproject(x, y, eta):
        u, _ = cg(lambda v: A_sr4(A_transpose(v)) + eta * v, y - A_sr4(x), tol=1e-6, maxiter=500)
        return x + A_transpose(u)

    def backproject(x, y_lr, eta=0.0):
        y = jnp.asarray(np.asarray(y_lr, np.float32)[None])
        return np.asarray(_backproject(jnp.asarray(x, jnp.float32)[None], y, jnp.float32(eta))[0])

    return backproject


def evaluate(index_rows, y_noisy, lr_all):
    import torch
    from iadrift.metrics import band_snr_db, blockiness, lpips_dist, psnr
    from iadrift.noisy import NOISE_FLOOR
    from iadrift.operators import bicubic_up, lambda_max_AAt, lr_loss

    torch.set_num_threads(4)
    backproject = make_backproject()
    lam_max = lambda_max_AAt()
    etas = {f"{f:g}": f * lam_max for f in ETA_FACTORS}
    print(f"lambda_max(A A^T) = {lam_max:.6f}; eta grid {etas}")

    def metrics(x, hr, i):
        return {"psnr": psnr(hr, x), "lpips": lpips_dist(x, hr), "blockiness": blockiness(x),
                "lr_loss_noisy": lr_loss(x, y_noisy[i]), "lr_err_clean": lr_loss(x, lr_all[i]),
                "band_snr_db": band_snr_db(x, hr)}

    per_image = {}
    for i in IMAGES:
        hr = data.hr_image(index_rows, i)
        outputs = {"bicubic": (bicubic_up(y_noisy[i]), {})}
        for tag in LAMBDAS:
            runs = {r: load_npz(run_path(i, tag, r)) for r in range(NUM_RESTARTS)}
            kept = min(runs, key=lambda r: (runs[r][0]["final_data"], r))
            info = {"kept_restart": kept, "runs": {str(r): runs[r][0] for r in runs}}
            outputs[f"CSGM lam {tag}"] = (runs[kept][1]["x_hat"], info)
        per_image[str(i)] = {}
        for name, (x, info) in outputs.items():
            entry = {"info": info, "no BP": metrics(x, hr, i)}
            for e_tag, eta in etas.items():
                entry[f"BP eta {e_tag}"] = metrics(backproject(x, y_noisy[i], eta), hr, i)
            per_image[str(i)][name] = entry
        print(f"image {i} evaluated", flush=True)

    def mean(name, variant, key):
        return float(np.mean([per_image[str(i)][name][variant][key] for i in IMAGES]))

    eta_star = max(etas, key=lambda e: mean("bicubic", f"BP eta {e}", "psnr"))
    variant = f"BP eta {eta_star}"
    lam_star = max(LAMBDAS, key=lambda t: (round(mean(f"CSGM lam {t}", variant, "psnr"), 2),
                                           -mean(f"CSGM lam {t}", variant, "lpips")))
    names = ["bicubic"] + [f"CSGM lam {t}" for t in LAMBDAS]
    variants = ["no BP"] + [f"BP eta {e}" for e in etas]
    means = {n: {v: {k: mean(n, v, k) for k in ("psnr", "lpips", "blockiness", "lr_loss_noisy", "lr_err_clean")}
                 for v in variants} for n in names}
    result = {
        "sigma": SIGMA, "noise_floor": NOISE_FLOOR, "lambda_max_AAt": lam_max, "etas": etas,
        "lambdas": LAMBDAS, "images": IMAGES, "eta_star": eta_star, "lambda_star": lam_star,
        "selection_rule": SELECTION_RULE,
        "means": means, "per_image": per_image,
    }
    print(f"\nMean over {len(IMAGES)} development images: PSNR dB / LPIPS")
    print(f"{'':16}" + "".join(f"{v:>18}" for v in variants))
    for n in names:
        print(f"{n:16}" + "".join(f"{means[n][v]['psnr']:>11.2f} / {means[n][v]['lpips']:.3f}" for v in variants))
    print(f"\nSelected: eta* = {eta_star} x lambda_max = {etas[eta_star]:.6f}; lambda* = {lam_star} "
          f"({LAMBDAS[lam_star]:.6f})")
    return result


def max_abs_diff(a, b, path=""):
    """(largest numeric difference, where); strings, ints and keys must match exactly."""
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        return max((max_abs_diff(a[k], b[k], f"{path}/{k}") for k in a), default=(0.0, path))
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        return max((max_abs_diff(u, v, f"{path}[{n}]") for n, (u, v) in enumerate(zip(a, b))), default=(0.0, path))
    if isinstance(a, str) or isinstance(a, bool) or a is None:
        assert a == b, (path, a, b)
        return 0.0, path
    return abs(float(a) - float(b)), path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-runs", type=int, default=10, help="runs per call (the sweep is resumable)")
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; runs and writes nothing")
    parser.add_argument("--evaluate", action="store_true", help="BP grid, metrics, LPIPS, selection -> grid.json")
    parser.add_argument("--check", action="store_true", help="with --evaluate: CPU, compare with grid.json; no writes")
    parser.add_argument("--smoke", action="store_true", help="GPU: 100-step checks against the cache; writes nothing")
    args = parser.parse_args()
    if args.check and not args.evaluate:
        parser.error("--check is a mode of --evaluate")

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train, test = data.load_split(labels)
    assert set(IMAGES) <= set(train.tolist()) and not set(IMAGES) & set(test.tolist()), "development images only"
    lr_all = data.load_measurements()
    if args.dry_run:
        from iadrift.noisy import NOISY_Y_NPZ
        done = sum(run_path(*p).exists() for p in plan())
        print("DRY RUN: nothing is run or written.")
        print(f"sigma {SIGMA:.5f}, noise floor {64 * 64 * 3 * SIGMA ** 2:.2f}; lambdas {LAMBDAS}; eta factors "
              f"{ETA_FACTORS}\nimages {IMAGES}; runs {len(plan())}, saved {done}; grid.json "
              f"{'present' if (OUT_DIR / 'grid.json').exists() else 'missing'}; noisy y file "
              f"{'exists' if (paths.DATA_DIR / NOISY_Y_NPZ).exists() else 'will be created'}")
        return

    y_noisy = load_noisy_y(labels, lr_all)
    if args.evaluate:
        result = evaluate(index_rows, y_noisy, lr_all)
        if args.check:
            saved = json.loads((OUT_DIR / "grid.json").read_text())
            same_choice = (result["eta_star"], result["lambda_star"]) == (saved["eta_star"], saved["lambda_star"])
            diff, where = max_abs_diff(json.loads(json.dumps(result)), saved)
            print(f"grid.json recomputed on the CPU: same selection {same_choice}; "
                  f"max |difference| {diff:.2e} ({where})")
            ok = same_choice and diff < 1e-3
            print("CHECK PASSED" if ok else "CHECK FAILED")
            raise SystemExit(0 if ok else 1)
        atomic_write_bytes(OUT_DIR / "grid.json", json.dumps(result, indent=1).encode())
        print(f"-> {OUT_DIR / 'grid.json'}")
        return

    from iadrift.csgm import CSGM
    from iadrift.generator import load_generator
    from iadrift.noisy import NoisyCSGM

    ncsgm = NoisyCSGM(CSGM(load_generator()))
    if args.smoke:
        smoke(index_rows, labels, y_noisy, ncsgm)
    sweep(args, index_rows, labels, y_noisy, ncsgm)


if __name__ == "__main__":
    main()
