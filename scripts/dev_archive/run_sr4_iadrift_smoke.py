"""IADrift smoke test on one image: oracle vs wrong label, from saved CSGM solutions.

For each condition, start from the saved CSGM solution (eps_CSGM, noise_labels,
same class label) and jointly minimize ||y - A f_theta(eps)||^2 over the
generator weights theta (init: pretrained params) and eps, with separate Adam
optimizers. Final CSGM and IADrift metrics both use the eager generator path,
the same evaluation as the saved CSGM numbers.

Usage:
    python scripts/run_sr4_iadrift_smoke.py                     # image 1704, n1 CSGM results
    python scripts/run_sr4_iadrift_smoke.py --steps 2000 --lr-theta 3e-6

Outputs (in DATA_DIR/iadrift_smoke/<tag>/): summary.json, one .npz per
condition, and figures in figures/iadrift_smoke/<tag>/.
"""

import os

# The default allocator hangs on the full theta-gradient under WSL (see notebook).
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import io
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuses the benchmark's model, operator, data and IO helpers (loads the model).
from run_sr4_csgm_test import (
    A_sr4, DATA_DIR, atomic_write_bytes, bicubic_up, jax, jnp, load_hr256,
    load_index, model, np, optax, params, psnr,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
CLASS_INDEX = json.loads((DATA_DIR / "imagenet_class_index.json").read_text())
TRACE_EVERY = 50
PSNR_EVERY = 250


def class_name(label):
    return CLASS_INDEX[str(int(label))][1].replace("_", " ")


def generate(theta, eps, label, noise_labels):
    condition = model.apply(
        {"params": theta}, jnp.asarray(label, dtype=jnp.int32).reshape(1),
        1.0, noise_labels, method=model.c_cfg_noise_to_cond,
    )
    x_hat = model.apply({"params": theta}, eps, condition, True, method=model.generate_image)
    return x_hat.astype(jnp.float32)


def iadrift_loss(theta, eps, y, label, noise_labels):
    return jnp.sum((A_sr4(generate(theta, eps, label, noise_labels)) - y) ** 2)


generate_jit = jax.jit(generate)


def make_joint_step(lr_theta, lr_eps):
    opt_theta, opt_eps = optax.adam(lr_theta), optax.adam(lr_eps)

    def step(theta, eps, state_theta, state_eps, y, label, noise_labels):
        loss, (g_theta, g_eps) = jax.value_and_grad(iadrift_loss, argnums=(0, 1))(
            theta, eps, y, label, noise_labels
        )
        updates, state_theta = opt_theta.update(g_theta, state_theta, theta)
        theta = optax.apply_updates(theta, updates)
        updates, state_eps = opt_eps.update(g_eps, state_eps, eps)
        eps = optax.apply_updates(eps, updates)
        return theta, eps, state_theta, state_eps, loss

    # theta, eps and both optimizer states are donated: updated in place on the GPU.
    return opt_theta, opt_eps, jax.jit(step, donate_argnums=(0, 1, 2, 3))


def eager_metrics(theta, eps, label, noise_labels, y, hr):
    """Final evaluation without jit, as in the CSGM benchmark."""
    x = generate(theta, eps, label, noise_labels)
    loss = float(jnp.sum((A_sr4(x) - y) ** 2))
    x = np.asarray(x[0])
    return x, loss, psnr(hr, x)


def load_csgm(results_path, image_index, condition):
    rows = [json.loads(line) for line in results_path.open() if line.strip()]
    rows = [r for r in rows if r["image_index"] == image_index and r["condition"] == condition]
    if not rows or len(rows) != rows[0]["num_restarts"]:
        raise SystemExit(f"Image {image_index} ({condition}) missing or unfinished in {results_path.name}")
    best = min(rows, key=lambda r: (r["final_loss"], r["restart"]))
    with np.load(Path(str(results_path.with_suffix("")) + "_arrays") / best["array_file"]) as arrays:
        return best, {k: arrays[k] for k in arrays.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=int, default=1704)
    parser.add_argument("--csgm-num-images", type=int, default=1, help="which v2 CSGM results file to start from")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--lr-theta", type=float, default=1e-5)
    parser.add_argument("--lr-eps", type=float, default=1e-3)
    args = parser.parse_args()

    matches = sorted(DATA_DIR.glob(f"phase1_sr4_csgm_label_test_v2_n{args.csgm_num_images}_*.jsonl"))
    if len(matches) != 1:
        raise SystemExit(f"Expected one CSGM results file, found {matches}")
    results_path = matches[0]

    index_rows = load_index()
    i = args.image
    hr = load_hr256(DATA_DIR / index_rows[i]["relative_path"])
    lr_all = np.load(DATA_DIR / "imageNet_x4.npy", mmap_mode="r")
    y = jnp.asarray(lr_all[i:i + 1], dtype=jnp.float32)
    bicubic = bicubic_up(lr_all[i])

    tag = f"img{i:05d}_steps{args.steps}_lrt{args.lr_theta:g}_lre{args.lr_eps:g}"
    out_dir = DATA_DIR / "iadrift_smoke" / tag
    fig_dir = FIGURES_DIR / "iadrift_smoke" / tag
    fig_dir.mkdir(parents=True, exist_ok=True)

    opt_theta, opt_eps, joint_step = make_joint_step(args.lr_theta, args.lr_eps)
    theta0_norm = float(optax.global_norm(params))
    summary = {"image_index": i, "true_label": int(index_rows[i]["label"]), "csgm_results": results_path.name,
               "steps": args.steps, "lr_theta": args.lr_theta, "lr_eps": args.lr_eps,
               "bicubic_psnr": psnr(hr, bicubic), "conditions": {}}
    images = {}

    for condition in ("oracle", "wrong"):
        row, csgm = load_csgm(results_path, i, condition)
        label = row["class_label"]
        noise_labels = jnp.asarray(csgm["noise_labels"])
        eps = jnp.asarray(csgm["eps"])
        theta = jax.device_put(params)  # fresh per-image copy of the pretrained weights
        state_theta, state_eps = opt_theta.init(theta), opt_eps.init(eps)

        # CSGM metrics re-evaluated with the eager path (should match the saved row).
        _, csgm_loss, csgm_psnr = eager_metrics(theta, eps, label, noise_labels, y, hr)

        trace_steps, trace_loss, psnr_steps, psnr_trace = [], [], [0], [csgm_psnr]
        print(f"\n[{condition}] class {label} ({class_name(label)}); CSGM restart {row['restart']}: "
              f"saved loss {row['final_loss']:.3f}, eager loss {csgm_loss:.3f}, PSNR {csgm_psnr:.2f} dB", flush=True)
        start = perf_counter()
        for step in range(1, args.steps + 1):
            theta, eps, state_theta, state_eps, loss = joint_step(
                theta, eps, state_theta, state_eps, y, label, noise_labels
            )
            if step == 1:
                jax.block_until_ready(loss)
                compile_and_first = perf_counter() - start
                start = perf_counter()
            if step % TRACE_EVERY == 0 or step == 1:
                trace_steps.append(step - 1)
                trace_loss.append(float(loss))
            if step % PSNR_EVERY == 0:
                x = np.asarray(generate_jit(theta, eps, label, noise_labels)[0])
                psnr_steps.append(step)
                psnr_trace.append(psnr(hr, x))
                print(f"  step {step}: loss (before update) {float(loss):.3f}, HR PSNR {psnr_trace[-1]:.2f} dB", flush=True)
        jax.block_until_ready(theta)
        run_seconds = perf_counter() - start

        x_iad, iad_loss, iad_psnr = eager_metrics(theta, eps, label, noise_labels, y, hr)
        theta_change = float(optax.global_norm(jax.tree_util.tree_map(lambda a, b: a - b, theta, params)))
        result = {
            "class_label": label, "class_name": class_name(label), "csgm_restart": row["restart"],
            "csgm_saved_loss": row["final_loss"], "csgm_saved_psnr": row["psnr"],
            "csgm_eager_loss": csgm_loss, "csgm_eager_psnr": csgm_psnr,
            "iadrift_jit_loss_start": trace_loss[0], "iadrift_jit_loss_last": trace_loss[-1],
            "iadrift_eager_loss": iad_loss, "iadrift_eager_psnr": iad_psnr,
            "iadrift_psnr_clipped": psnr(hr, np.clip(x_iad, -1, 1)),
            "theta_relative_change": theta_change / theta0_norm,
            "final_eps_rms": float(jnp.sqrt(jnp.mean(eps ** 2))),
            "compile_and_first_step_seconds": compile_and_first,
            "seconds_per_step": run_seconds / max(args.steps - 1, 1),
        }
        summary["conditions"][condition] = result
        images[condition] = (csgm["x_hat"], x_iad)

        buffer = io.BytesIO()
        np.savez_compressed(buffer, x_hat=x_iad, eps=np.asarray(eps), noise_labels=np.asarray(noise_labels),
                            trace_steps=np.asarray(trace_steps), trace_loss=np.asarray(trace_loss),
                            psnr_steps=np.asarray(psnr_steps), psnr_trace=np.asarray(psnr_trace))
        atomic_write_bytes(out_dir / f"{condition}.npz", buffer.getvalue())
        print(f"[{condition}] IADrift: eager loss {iad_loss:.3f}, PSNR {iad_psnr:.2f} dB "
              f"(CSGM {csgm_psnr:.2f}); |theta - theta0|/|theta0| = {result['theta_relative_change']:.2e}; "
              f"{result['seconds_per_step'] * 1000:.0f} ms/step", flush=True)
        del theta, state_theta, state_eps

        summary["conditions"][condition]["_trace"] = (trace_steps, trace_loss, psnr_steps, psnr_trace)

    traces = {c: summary["conditions"][c].pop("_trace") for c in ("oracle", "wrong")}
    atomic_write_bytes(out_dir / "summary.json", (json.dumps(summary, indent=2) + "\n").encode())

    # Figure 1: images. Columns: reference | wrong label | correct label.
    def show(ax, x, title):
        ax.imshow((np.clip(x, -1, 1) + 1) / 2)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    c_o, c_w = summary["conditions"]["oracle"], summary["conditions"]["wrong"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9.8))
    show(axes[0, 0], hr, f"Original HR\n{class_name(summary['true_label'])}")
    show(axes[1, 0], bicubic, f"Bicubic upsample\nPSNR {summary['bicubic_psnr']:.2f} dB")
    for col, (cond, c) in ((1, ("wrong", c_w)), (2, ("oracle", c_o))):
        what = "wrong label" if cond == "wrong" else "correct label"
        show(axes[0, col], images[cond][0],
             f"CSGM, {what}: {c['class_name']}\nPSNR {c['csgm_eager_psnr']:.2f} dB, LR loss {c['csgm_eager_loss']:.1f}")
        show(axes[1, col], images[cond][1],
             f"IADrift, {what}: {c['class_name']}\nPSNR {c['iadrift_eager_psnr']:.2f} dB, LR loss {c['iadrift_eager_loss']:.1f}")
    fig.suptitle(f"Test image {i}  |  SR x4  |  IADrift: {args.steps:,} joint steps, "
                 f"lr θ {args.lr_theta:g}, lr ε {args.lr_eps:g}  |  metrics: eager evaluation", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9, wspace=0.04, hspace=0.2)
    fig.savefig(fig_dir / "images.png", dpi=110)
    plt.close(fig)

    # Figure 2: optimization traces.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for cond, style in (("oracle", "-"), ("wrong", "--")):
        ts, tl, ps, pt = traces[cond]
        name = f"{'correct' if cond == 'oracle' else 'wrong'}: {summary['conditions'][cond]['class_name']}"
        ax1.plot(ts, tl, style, label=name)
        ax2.plot(ps, pt, style, marker="o", label=name)
    ax1.set(xlabel="IADrift step", ylabel="LR loss (jit, before update)", yscale="log", title="Measurement loss")
    ax2.axhline(summary["bicubic_psnr"], color="gray", linestyle=":", label="bicubic")
    ax2.set(xlabel="IADrift step", ylabel="HR PSNR (dB)", title="HR PSNR (evaluation only)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "traces.png", dpi=110)
    plt.close(fig)
    print(f"\nSaved {out_dir / 'summary.json'} and figures in {fig_dir}")


if __name__ == "__main__":
    main()
