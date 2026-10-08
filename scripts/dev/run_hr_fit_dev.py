"""IADrift fitted to HR, A = identity (development, 4 images; notebook 02, the IADrift-HR rows and curves).

Diagnostic (uses HR as the target): can adapting the weights close the frozen generator's representation gap?
Start from the R1 solution (CSGM fitted to HR, the restart with the lower final HR loss in
representation_error_dev_v1/, scripts/dev/run_representation_dev.py) and jointly fit weights and noise:

    (theta_hat, eps_hat) = argmin_{theta, eps} ||HR - G_theta(eps; z, c)||^2

theta from the pretrained weights (all 241 tensors), two Adam optimizers (theta lr 1e-6 and 1e-5, eps lr 1e-3, the SR
IADrift learning rates), 2,000 steps, z and the correct class fixed. Images 8117, 3939, 403, 347. Every 50 steps (and
at step 0) the compiled output is scored (HR loss, PSNR, LPIPS, LR loss, band SNR): evaluation only. The final output
comes from the eager generator path.

Ported from scripts/run_hr_fit_iadrift_dev.py (the loss ||G_theta(eps) - HR||^2 is that script's hr_fit_loss and
step, copied; the generator call is iadrift.generator.Generator.generate_theta). theta gradients: platform allocator.

Usage:
    python scripts/dev/run_hr_fit_dev.py --dry-run     # plan and cache status; runs and writes nothing (CPU)
    python scripts/dev/run_hr_fit_dev.py --smoke       # rerun 100 steps of one saved run, compare the trace
    python scripts/dev/run_hr_fit_dev.py               # run what is missing (existing files are kept)

Outputs (IADRIFT_DATA/hr_fit_iadrift_dev_v1/): img#####_lrt{lr}_lre0.001_steps2000.npz with x (eager output),
eps, and result_json (settings, r1_restart, seconds, the every-50-steps trace).
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

DEV_IMAGES = list(data.DEV_IMAGES_4)
SETTINGS = [(1e-6, 1e-3), (1e-5, 1e-3)]  # (theta lr, eps lr)
STEPS = 2_000
TRACE_EVERY = 50
R1_DIR = paths.DATA_DIR / "representation_error_dev_v1"
OUT_DIR = paths.DATA_DIR / "hr_fit_iadrift_dev_v1"
SMOKE = {"image_index": 8117, "lr_theta": 1e-5, "lr_eps": 1e-3, "steps": 100}


def path_of(i, lr_theta, lr_eps):
    return OUT_DIR / f"img{i:05d}_lrt{lr_theta:g}_lre{lr_eps:g}_steps{STEPS}.npz"


def r1_start(i):
    """Lowest-final-HR-loss R1 restart: (restart, eps, noise_labels)."""
    runs = []
    for r in (0, 1):
        metrics, arrays = load_npz(R1_DIR / f"R1_img{i:05d}_r{r}.npz")
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
    plan = [(lt, le, i) for lt, le in SETTINGS for i in DEV_IMAGES]

    if args.dry_run:
        print("DRY RUN: nothing is run or written.")
        for lt, le, i in plan:
            print(f"  theta lr {lt:g}, eps lr {le:g}, image {i} (R1 start r{r1_start(i)[0]}): "
                  f"{'cached' if path_of(i, lt, le).exists() else 'MISSING'}")
        print(f"Outputs present: {sum(path_of(i, lt, le).exists() for lt, le, i in plan)} / {len(plan)} in {OUT_DIR}")
        return

    import jax
    import jax.numpy as jnp
    import optax
    import torch

    from iadrift.adaptation import Adaptation
    from iadrift.generator import load_generator
    from iadrift.metrics import band_snr_db, lpips_dist, psnr
    from iadrift.operators import A_sr4

    torch.set_num_threads(4)  # as the original (LPIPS on the CPU)
    ad = Adaptation(load_generator())
    generate, generate_jit = ad.gen.generate_theta, ad.generate_jit
    lr_all = data.load_measurements()

    def hr_fit_loss(theta, eps, hr, label, noise_labels):
        return jnp.sum((generate(theta, eps, label, noise_labels) - hr) ** 2)

    def make_joint_hr_step(lr_theta, lr_eps):
        opt_theta, opt_eps = optax.adam(lr_theta), optax.adam(lr_eps)

        def step(theta, eps, state_theta, state_eps, hr, label, noise_labels):
            loss, (g_theta, g_eps) = jax.value_and_grad(hr_fit_loss, argnums=(0, 1))(
                theta, eps, hr, label, noise_labels)
            updates, state_theta = opt_theta.update(g_theta, state_theta, theta)
            theta = optax.apply_updates(theta, updates)
            updates, state_eps = opt_eps.update(g_eps, state_eps, eps)
            eps = optax.apply_updates(eps, updates)
            return theta, eps, state_theta, state_eps, loss

        return opt_theta, opt_eps, jax.jit(step, donate_argnums=(0, 1, 2, 3))

    def run(i, lr_theta, lr_eps, steps, step_fns):
        opt_theta, opt_eps, step_fn = step_fns
        label, hr = int(labels[i]), data.hr_image(index_rows, i)
        y = jnp.asarray(lr_all[i:i + 1], jnp.float32)
        restart, eps, noise_labels = r1_start(i)
        theta = ad.fresh_weights()  # params itself is never donated
        eps, noise_labels = jnp.asarray(eps), jnp.asarray(noise_labels)
        state_theta, state_eps = opt_theta.init(theta), opt_eps.init(eps)
        hr_b = jnp.asarray(hr, jnp.float32)[None]
        trace = {k: [] for k in ("step", "hr_loss", "psnr", "lpips", "lr_loss", "low", "mid", "high", "very high")}

        def record(step):
            x = np.asarray(generate_jit(theta, eps, label, noise_labels)[0])
            trace["step"].append(step)
            trace["hr_loss"].append(float(np.sum((x - hr) ** 2)))
            trace["psnr"].append(psnr(hr, x))
            trace["lpips"].append(lpips_dist(x, hr))
            trace["lr_loss"].append(float(jnp.sum((A_sr4(jnp.asarray(x)[None]) - y) ** 2)))
            for band, value in band_snr_db(x, hr).items():
                trace[band].append(value)

        start = perf_counter()
        record(0)
        for step in range(1, steps + 1):
            theta, eps, state_theta, state_eps, _ = step_fn(theta, eps, state_theta, state_eps, hr_b, label,
                                                            noise_labels)
            if step % TRACE_EVERY == 0:
                record(step)
        x, _, _ = ad.eager_metrics(theta, eps, label, noise_labels, y, hr)
        result = {"image_index": i, "label": label, "lr_theta": lr_theta, "lr_eps": lr_eps, "steps": steps,
                  "r1_restart": restart, "seconds": perf_counter() - start, "trace": trace}
        del theta, state_theta
        return result, x, np.asarray(eps, np.float32)

    if args.smoke:
        i, lt, le, k = SMOKE["image_index"], SMOKE["lr_theta"], SMOKE["lr_eps"], SMOKE["steps"]
        with np.load(path_of(i, lt, le)) as saved:
            saved_trace = json.loads(str(saved["result_json"]))["trace"]
        trace = run(i, lt, le, k, make_joint_hr_step(lt, le))[0]["trace"]
        n = len(trace["step"])
        diff = {key: max(abs(a - b) for a, b in zip(trace[key], saved_trace[key][:n])) for key in trace}
        ok = all(trace[key] == saved_trace[key][:n] for key in trace)
        print(f"[smoke] image {i}, theta lr {lt:g}: trace at steps {trace['step']} identical: {ok}; "
              f"max |difference| per field: {diff}")
        print("SMOKE PASSED" if ok else "SMOKE FAILED")
        raise SystemExit(0 if ok else 1)

    for lr_theta, lr_eps in SETTINGS:
        todo = [i for lt, le, i in plan if (lt, le) == (lr_theta, lr_eps) and not path_of(i, lt, le).exists()]
        if not todo:
            continue
        step_fns = make_joint_hr_step(lr_theta, lr_eps)
        for i in todo:
            result, x, eps = run(i, lr_theta, lr_eps, STEPS, step_fns)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, result_json=json.dumps(result), x=x, eps=eps)
            atomic_write_bytes(path_of(i, lr_theta, lr_eps), buffer.getvalue())
            t = result["trace"]
            print(f"theta lr {lr_theta:g}, image {i}: PSNR {t['psnr'][0]:.2f} -> {t['psnr'][-1]:.2f} dB, "
                  f"LPIPS {t['lpips'][0]:.3f} -> {t['lpips'][-1]:.3f} ({result['seconds']:.0f} s)", flush=True)
    print("ALL RUNS COMPLETE")


if __name__ == "__main__":
    main()
