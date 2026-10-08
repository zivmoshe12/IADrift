"""theta-only adaptation from CSGM (development, 4 images; notebook 02 "why adaptation fails", theta-only rows/curves).

Freeze the CSGM solution's (eps, z) and the class, and adapt only the generator weights:

    theta_hat = argmin_theta ||y - A G_theta(eps_CSGM; z, c)||^2,     x_hat = G_theta_hat(eps_CSGM; z, c)

(IADrift with the eps learning rate exactly zero: no eps optimizer at all). theta starts at the pretrained weights
(a fresh copy per run), Adam on all 241 tensors, 2,000 steps, theta lr 1e-6 and 1e-5, each from the same start.
Start: the correct-label restart of csgm_dev_5000steps_lr0.02/ with the lowest final LR loss
(scripts/dev/run_csgm_dev.py). Images: 8117, 3939, 403, 347. Every 50 steps (and at step 0) the compiled output is
scored against HR (LR loss, PSNR, band SNR): evaluation only. The final output comes from the eager generator path.

Ported from notebooks/05_theta_only_dev.ipynb cells 4 and 6 (make_theta_step = iadrift.adaptation.Adaptation
.make_theta_step; run_theta_only). theta gradients: platform allocator, as the original.

Usage:
    python scripts/dev/run_theta_only_dev.py --dry-run     # plan and cache status; runs and writes nothing (CPU)
    python scripts/dev/run_theta_only_dev.py --smoke       # rerun 100 steps of one saved run, compare the trace
    python scripts/dev/run_theta_only_dev.py               # run what is missing (existing files are kept)

Outputs (IADRIFT_DATA/theta_only_dev_v1/): img#####_lrt{lr}_steps2000.npz with x (the eager output at step 2,000)
and result_json (settings, seconds, csgm_restart, the every-50-steps trace).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

import os

if "--dry-run" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from iadrift.runtime import setup

setup(theta_gradients=True)  # gradients w.r.t. the generator weights

import argparse
import io
import json
from time import perf_counter

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import atomic_write_bytes, load_npz
from iadrift.metrics import BANDS, band_snr_db, psnr

OUT_DIR = paths.DATA_DIR / "theta_only_dev_v1"
CSGM_DIR = paths.DATA_DIR / "csgm_dev_5000steps_lr0.02"
DEV_IMAGES = list(data.DEV_IMAGES_4)
LR_THETAS = [1e-6, 1e-5]
STEPS = 2_000
EVAL_EVERY = 50
BAND_NAMES = [name for name, *_ in BANDS]
SMOKE = {"image_index": 8117, "lr_theta": 1e-5, "steps": 100}


def path_of(i, lr_theta):
    return OUT_DIR / f"img{i:05d}_lrt{lr_theta:g}_steps{STEPS}.npz"


def csgm_start(i, label):
    """The cached correct-label CSGM restart with the lowest final LR loss: (restart, eps, noise_labels)."""
    runs = []
    for r in range(2):
        metrics, arrays = load_npz(CSGM_DIR / f"img{i:05d}_oracle_r{r}.npz")
        assert metrics["class_label"] == label
        runs.append((metrics["final_loss"], r, arrays["eps"], arrays["noise_labels"]))
    _, r, eps, noise_labels = min(runs, key=lambda t: (t[0], t[1]))
    return r, eps, noise_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; run and write nothing")
    parser.add_argument("--smoke", action="store_true", help="rerun the start of one saved run; writes nothing")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    _, test_indices = data.load_split(labels)
    assert not np.any(np.isin(DEV_IMAGES, test_indices)), "development images only"
    plan = [(lt, i) for lt in LR_THETAS for i in DEV_IMAGES]

    if args.dry_run:
        print("DRY RUN: nothing is run or written.")
        for lt, i in plan:
            r, _, _ = csgm_start(i, int(labels[i]))
            state = "cached" if path_of(i, lt).exists() else "MISSING"
            print(f"  theta lr {lt:g}, image {i} (CSGM start r{r}): {state}")
        print(f"Outputs present: {sum(path_of(i, lt).exists() for lt, i in plan)} / {len(plan)} in {OUT_DIR}")
        return

    import jax.numpy as jnp

    from iadrift.adaptation import Adaptation
    from iadrift.generator import load_generator
    from iadrift.operators import A_sr4

    ad = Adaptation(load_generator())
    lr_all = data.load_measurements()

    def lr_loss(x, y):
        return float(jnp.sum((A_sr4(jnp.asarray(x, jnp.float32)[None]) - y) ** 2))

    def image(i):
        label = int(labels[i])
        r, eps, noise_labels = csgm_start(i, label)
        return {"label": label, "hr": data.hr_image(index_rows, i), "y": jnp.asarray(lr_all[i:i + 1], jnp.float32),
                "eps": jnp.asarray(eps), "noise_labels": jnp.asarray(noise_labels), "restart": r}

    def run_theta_only(im, step_fn, opt, steps):
        theta = ad.fresh_weights()  # the original params are never donated
        state = opt.init(theta)
        trace = {"step": [], "lr_loss": [], "psnr": [], **{b: [] for b in BAND_NAMES}}

        def record(step):
            x = np.asarray(ad.generate_jit(theta, im["eps"], im["label"], im["noise_labels"])[0])
            trace["step"].append(step)
            trace["lr_loss"].append(lr_loss(x, im["y"]))
            trace["psnr"].append(psnr(im["hr"], x))
            for b, value in band_snr_db(x, im["hr"]).items():
                trace[b].append(value)

        start = perf_counter()
        record(0)
        for step in range(1, steps + 1):
            theta, state, _ = step_fn(theta, state, im["eps"], im["y"], im["label"], im["noise_labels"])
            if step % EVAL_EVERY == 0:
                record(step)
        x, _, _ = ad.eager_metrics(theta, im["eps"], im["label"], im["noise_labels"], im["y"], im["hr"])
        del theta, state
        return trace, x, perf_counter() - start

    if args.smoke:
        i, lt, k = SMOKE["image_index"], SMOKE["lr_theta"], SMOKE["steps"]
        with np.load(path_of(i, lt)) as saved:
            saved_trace = json.loads(str(saved["result_json"]))["trace"]
        opt, step_fn = ad.make_theta_step(lt)
        trace, _, _ = run_theta_only(image(i), step_fn, opt, k)
        n = len(trace["step"])
        diff = {key: max(abs(a - b) for a, b in zip(trace[key], saved_trace[key][:n])) for key in trace}
        ok = all(trace[key] == saved_trace[key][:n] for key in trace)
        print(f"[smoke] image {i}, theta lr {lt:g}: trace at steps {trace['step']} identical: {ok}; "
              f"max |difference| per field: {diff}")
        print("SMOKE PASSED" if ok else "SMOKE FAILED")
        raise SystemExit(0 if ok else 1)

    images = {}
    for lt in LR_THETAS:
        todo = [i for i in DEV_IMAGES if not path_of(i, lt).exists()]
        if not todo:
            continue
        opt, step_fn = ad.make_theta_step(lt)
        for i in todo:
            im = images.setdefault(i, image(i))
            trace, x, seconds = run_theta_only(im, step_fn, opt, STEPS)
            result = {"image_index": i, "lr_theta": lt, "steps": STEPS, "seconds": seconds,
                      "csgm_restart": im["restart"], "trace": trace}
            buffer = io.BytesIO()
            np.savez_compressed(buffer, result_json=json.dumps(result), x=x)
            atomic_write_bytes(path_of(i, lt), buffer.getvalue())
            print(f"theta lr {lt:g}, image {i}: {seconds:.0f} s, PSNR {trace['psnr'][0]:.2f} -> "
                  f"{psnr(im['hr'], x):.2f} dB", flush=True)
    print(f"ALL RUNS COMPLETE: {OUT_DIR}")


if __name__ == "__main__":
    main()
