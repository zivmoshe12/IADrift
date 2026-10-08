"""IADrift learning-rate grid (development, 4 images; notebook 02 "why adaptation fails", IADrift rows and curves).

IADrift (IAGAN-style joint adaptation) from the CSGM solution, correct label:

    (theta_hat, eps_hat) = argmin_{theta, eps} ||y - A G_theta(eps; z, c)||^2,     x_hat = G_theta_hat(eps_hat; z, c)

theta starts at the pretrained weights (all 241 tensors), (eps, z) at the CSGM solution: the restart of
csgm_dev_5000steps_lr0.02/ with the lowest final LR loss (scripts/dev/run_csgm_dev.py). Two Adam optimizers,
1,000 steps, grid theta lr {1e-6, 3e-6, 1e-5} x eps lr {0 (eps frozen), 1e-3} = 6 settings x 4 development images
(8117, 3939, 403, 347). Every 50 steps (and at step 0) the compiled generator output is scored against HR (PSNR, LR
loss, band SNR): evaluation only, never used to stop or choose. Final output from the eager generator path, plus
"+ BP" (back-projection onto {x : A x = y}) and "+ fusion" (bicubic outside 0.0625-0.125 cycles/pixel). The setting
used everywhere later (theta lr 1e-5, eps lr 1e-3) was fixed before this grid; the grid shows the trade-off.

Ported from notebooks/02_iadrift_dev_tuning.ipynb cells 4 and 6 (run_iadrift; step = run_sr4_iadrift_smoke
.make_joint_step = iadrift.adaptation.Adaptation.make_joint_step). theta gradients: platform allocator, as the original.

Reproducibility: this grid was the project's first IADrift run, made before XLA's deterministic GPU ops were enabled
for weight-gradient runs (iadrift.runtime.setup). Without them the compiled programs pick between GPU kernels per
process, so the cached traces are one draw of that variation (its step-0 LR loss is 65.23; the theta-only cache, same
start, has 65.47). A rerun therefore cannot match this cache bit-exactly; --smoke checks agreement within the observed
run-to-run variation (PSNR within 0.1 dB, LR loss within 2% over the first 100 steps). All later weight-gradient runners
reproduce their caches bit-exactly.

Usage:
    python scripts/dev/run_iadrift_dev.py --dry-run     # plan and cache status; runs and writes nothing (CPU)
    python scripts/dev/run_iadrift_dev.py --smoke       # rerun 100 steps of one saved run, compare the trace; no writes
    python scripts/dev/run_iadrift_dev.py               # run what is missing (existing files are kept)

Outputs (IADRIFT_DATA/iadrift_dev_tuning_v1/): img#####_lrt{lr}_lre{lr}.npz with arrays "IADrift", "IADrift + BP",
"IADrift + fusion" and result_json (settings, the every-50-steps trace, metrics of the three outputs).
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

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import atomic_write_bytes, load_npz
from iadrift.metrics import BANDS, band_snr_db, fuse_with_bicubic, psnr

OUT_DIR = paths.DATA_DIR / "iadrift_dev_tuning_v1"
CSGM_DIR = paths.DATA_DIR / "csgm_dev_5000steps_lr0.02"
DEV_IMAGES = list(data.DEV_IMAGES_4)
CONFIGS = [(lr_theta, lr_eps) for lr_theta in (1e-6, 3e-6, 1e-5) for lr_eps in (0.0, 1e-3)]
STEPS = 1_000
EVAL_EVERY = 50
BAND_NAMES = [name for name, *_ in BANDS]
OUTPUTS = ("IADrift", "IADrift + BP", "IADrift + fusion")
SMOKE = {"image_index": 8117, "lr_theta": 1e-5, "lr_eps": 1e-3, "steps": 100}


def path_of(i, lr_theta, lr_eps):
    return OUT_DIR / f"img{i:05d}_lrt{lr_theta:g}_lre{lr_eps:g}.npz"


def csgm_start(i, label):
    """The cached correct-label CSGM restart with the lowest final LR loss: (restart, eps, noise_labels)."""
    runs = []
    for r in range(2):
        metrics, arrays = load_npz(CSGM_DIR / f"img{i:05d}_oracle_r{r}.npz")
        assert metrics["class_label"] == label
        runs.append((metrics["final_loss"], r, arrays["eps"], arrays["noise_labels"]))
    _, r, eps, noise_labels = min(runs, key=lambda t: (t[0], t[1]))
    return r, eps, noise_labels


class DevIADrift:
    def __init__(self, gen):
        import jax.numpy as jnp

        from iadrift.adaptation import Adaptation
        from iadrift.operators import A_sr4

        self.ad, self.jnp, self.A_sr4 = Adaptation(gen), jnp, A_sr4

    def evaluate(self, x, hr, y):
        """Notebook 02's metrics; y is the [1, 64, 64, 3] device array."""
        jnp = self.jnp
        lr_loss = float(jnp.sum((self.A_sr4(jnp.asarray(x, jnp.float32)[None]) - y) ** 2))
        return {"psnr": psnr(hr, x), "lr_loss": lr_loss, "band_snr_db": band_snr_db(x, hr)}

    def adapt(self, im, lr_theta, lr_eps, steps, step_fn=None):
        """IADrift from the CSGM start; returns (trace, eager output)."""
        ad, jnp = self.ad, self.jnp
        opt_theta, opt_eps, joint_step = step_fn or ad.make_joint_step(lr_theta, lr_eps)
        theta = ad.fresh_weights()  # the original used jax.device_put(params); same values
        eps, noise_labels = jnp.asarray(im["eps"]), jnp.asarray(im["noise_labels"])
        state_theta, state_eps = opt_theta.init(theta), opt_eps.init(eps)
        trace = {"step": [], "psnr": [], "lr_loss": [], **{b: [] for b in BAND_NAMES}}

        def record(step):
            m = self.evaluate(np.asarray(ad.generate_jit(theta, eps, im["label"], noise_labels)[0]), im["hr"], im["y"])
            trace["step"].append(step)
            trace["psnr"].append(m["psnr"])
            trace["lr_loss"].append(m["lr_loss"])
            for b in BAND_NAMES:
                trace[b].append(m["band_snr_db"][b])

        record(0)
        for step in range(1, steps + 1):
            theta, eps, state_theta, state_eps, _ = joint_step(
                theta, eps, state_theta, state_eps, im["y"], im["label"], noise_labels)
            if step % EVAL_EVERY == 0:
                record(step)
        x, _, _ = ad.eager_metrics(theta, eps, im["label"], noise_labels, im["y"], im["hr"])
        del theta, state_theta, state_eps
        return trace, x


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; run and write nothing")
    parser.add_argument("--smoke", action="store_true", help="rerun the start of one saved run; writes nothing")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    _, test_indices = data.load_split(labels)
    assert not np.any(np.isin(DEV_IMAGES, test_indices)), "development images only"
    plan = [(lt, le, i) for lt, le in CONFIGS for i in DEV_IMAGES]

    if args.dry_run:
        print("DRY RUN: nothing is run or written.")
        for i in DEV_IMAGES:
            r, _, _ = csgm_start(i, int(labels[i]))
            print(f"  image {i}: CSGM start = restart r{r} (lowest final LR loss)")
        for lt, le, i in plan:
            state = "cached" if path_of(i, lt, le).exists() else "MISSING"
            print(f"  theta lr {lt:g}, eps lr {le:g}, image {i}: {state}")
        print(f"Outputs present: {sum(path_of(i, lt, le).exists() for lt, le, i in plan)} / {len(plan)} in {OUT_DIR}")
        return

    import jax.numpy as jnp

    from iadrift.generator import load_generator
    from iadrift.operators import backproject, bicubic_up

    runner = DevIADrift(load_generator())
    lr_all = data.load_measurements()

    def image(i):
        label = int(labels[i])
        r, eps, noise_labels = csgm_start(i, label)
        return {"label": label, "hr": data.hr_image(index_rows, i), "y": jnp.asarray(lr_all[i:i + 1], jnp.float32),
                "eps": eps, "noise_labels": noise_labels, "bicubic": bicubic_up(lr_all[i])}

    if args.smoke:
        i, lt, le, k = SMOKE["image_index"], SMOKE["lr_theta"], SMOKE["lr_eps"], SMOKE["steps"]
        with np.load(path_of(i, lt, le)) as saved:
            saved_trace = json.loads(str(saved["result_json"]))["trace"]
        trace, _ = runner.adapt(image(i), lt, le, k)
        n = len(trace["step"])
        diff = {key: max(abs(a - b) for a, b in zip(trace[key], saved_trace[key][:n])) for key in trace}
        exact = all(trace[key] == saved_trace[key][:n] for key in trace)
        rel_loss = max(abs(a - b) / b for a, b in zip(trace["lr_loss"], saved_trace["lr_loss"][:n]))
        ok = exact or (diff["psnr"] <= 0.1 and rel_loss <= 0.02)  # see "Reproducibility" in the docstring
        print(f"[smoke] image {i}, theta lr {lt:g}, eps lr {le:g}: trace at steps {trace['step']} identical: {exact}; "
              f"max |difference| per field: {diff}; max relative LR-loss difference {rel_loss:.4f}")
        print(("SMOKE PASSED" + ("" if exact else " (within tolerance: PSNR <= 0.1 dB, LR loss <= 2%)")) if ok
              else "SMOKE FAILED")
        raise SystemExit(0 if ok else 1)

    images = {}
    for lt, le in CONFIGS:
        todo = [i for i in DEV_IMAGES if not path_of(i, lt, le).exists()]
        if not todo:
            continue
        step_fn = runner.ad.make_joint_step(lt, le)
        for i in todo:
            im = images.setdefault(i, image(i))
            trace, x = runner.adapt(im, lt, le, STEPS, step_fn)
            outputs = {"IADrift": x, "IADrift + BP": backproject(x, lr_all[i]),
                       "IADrift + fusion": fuse_with_bicubic(x, im["bicubic"])}
            result = {"image_index": i, "lr_theta": lt, "lr_eps": le, "steps": STEPS, "trace": trace,
                      "metrics": {name: runner.evaluate(out, im["hr"], im["y"]) for name, out in outputs.items()}}
            buffer = io.BytesIO()
            np.savez_compressed(buffer, result_json=json.dumps(result), **outputs)
            atomic_write_bytes(path_of(i, lt, le), buffer.getvalue())
            print(f"theta lr {lt:g}, eps lr {le:g}, image {i}: PSNR " + ", ".join(
                f"{n} {result['metrics'][n]['psnr']:.2f}" for n in OUTPUTS) + " dB", flush=True)
    print(f"ALL RUNS COMPLETE: {OUT_DIR}")


if __name__ == "__main__":
    main()
