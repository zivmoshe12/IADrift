"""IADrift development runs: CSGM -> IADrift on development images, oracle vs wrong label.

Development images only: drawn (DEV_SEED) from the 100-image pool of the
development sweep and checked to be in the development split, not the test split.

Per image and condition (oracle / wrong label):
  1. CSGM, frozen config (lr 0.02, 5,000 steps, 2 restarts), restart chosen by
     lowest LR loss. Cached in DATA_DIR/csgm_dev_5000steps_lr0.02/ for reuse.
  2. IADrift from that solution: joint Adam over (theta, eps).
  3. Metrics for bicubic, CSGM and IADrift: LR loss, HR PSNR and band SNR per
     frequency band (freq_metrics.py), all from the eager evaluation path.

Usage:
    python scripts/run_sr4_iadrift_dev.py --num-images 4
    python scripts/run_sr4_iadrift_dev.py --images 9588 1234 --steps 500 --lr-theta 3e-6

Already-finished images in the same output folder are skipped (resumable).
"""

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # see run_sr4_iadrift_smoke.py

import argparse
import io
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from freq_metrics import BANDS, LR_NYQUIST, band_snr_db, radial_snr_db
from run_sr4_csgm_test import FROZEN_CONFIG, load_split, reconstruct, wrong_label_for
from run_sr4_iadrift_smoke import (
    DATA_DIR, FIGURES_DIR, atomic_write_bytes, bicubic_up, class_name, eager_metrics,
    generate_jit, jax, jnp, load_hr256, load_index, make_joint_step, np, optax, params, psnr,
)

DEV_SEED = 2030
DEV_POOL_PATH = DATA_DIR / "phase1_sr4_oracle_moments_dev_v1_6b22675bd557.jsonl"
CSGM_CACHE = DATA_DIR / f"csgm_dev_{FROZEN_CONFIG['num_steps']}steps_lr{FROZEN_CONFIG['learning_rate']:g}"
PSNR_EVERY = 250
TRACE_EVERY = 50
METHODS = ("bicubic", "CSGM oracle", "IADrift oracle", "CSGM wrong", "IADrift wrong")


def pick_dev_images(num_images, labels, train_indices, test_indices):
    pool = sorted({json.loads(line)["image_index"] for line in DEV_POOL_PATH.open() if line.strip()})
    chosen = np.random.default_rng(DEV_SEED).choice(pool, size=num_images, replace=False)
    return [int(i) for i in chosen]


def check_dev(images, train_indices, test_indices):
    bad = [i for i in images if i not in set(train_indices.tolist()) or i in set(test_indices.tolist())]
    if bad:
        raise SystemExit(f"Not development images: {bad}. Use development images for IADrift development.")


def csgm_cached(image_index, condition, restart, label, y_lr, hr):
    """Run (or load) one CSGM restart; returns (metrics, arrays)."""
    path = CSGM_CACHE / f"img{image_index:05d}_{condition}_r{restart}.npz"
    if path.exists():
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files if k != "metrics_json"}
            metrics = json.loads(str(data["metrics_json"]))
        if metrics["class_label"] == label:
            return metrics, arrays
    metrics, arrays = reconstruct(image_index, restart, label, y_lr, hr)
    metrics["class_label"] = label
    buffer = io.BytesIO()
    np.savez_compressed(buffer, metrics_json=json.dumps(metrics), **arrays)
    atomic_write_bytes(path, buffer.getvalue())
    return metrics, arrays


def run_iadrift(start_eps, noise_labels, label, y, hr, joint_step, opt_theta, opt_eps, steps):
    theta = jax.device_put(params)  # fresh copy of the pretrained weights
    eps = jnp.asarray(start_eps)
    state_theta, state_eps = opt_theta.init(theta), opt_eps.init(eps)
    trace_steps, trace_loss, psnr_steps, psnr_trace = [], [], [], []
    start = perf_counter()
    for step in range(1, steps + 1):
        theta, eps, state_theta, state_eps, loss = joint_step(
            theta, eps, state_theta, state_eps, y, label, noise_labels
        )
        if step == 1 or step % TRACE_EVERY == 0:
            trace_steps.append(step - 1)
            trace_loss.append(float(loss))
        if step % PSNR_EVERY == 0:
            psnr_steps.append(step)
            psnr_trace.append(psnr(hr, np.asarray(generate_jit(theta, eps, label, noise_labels)[0])))
    jax.block_until_ready(theta)
    seconds = perf_counter() - start
    x, loss, value = eager_metrics(theta, eps, label, noise_labels, y, hr)
    change = float(optax.global_norm(jax.tree_util.tree_map(lambda a, b: a - b, theta, params)))
    info = {
        "eager_loss": loss, "psnr": value, "seconds": seconds,
        "theta_relative_change": change / float(optax.global_norm(params)),
        "trace_steps": trace_steps, "trace_loss": trace_loss,
        "psnr_steps": psnr_steps, "psnr_trace": psnr_trace,
    }
    return x, np.asarray(eps), info


def plot_image(path, i, true_label, hr, recon, rows):
    def show(ax, x, title):
        ax.imshow((np.clip(x, -1, 1) + 1) / 2)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def label_of(method):
        r = rows[method]
        bands = f"SNR low/mid/high {r['band_snr_db']['low']:.1f}/{r['band_snr_db']['mid']:.1f}/{r['band_snr_db']['high']:.1f} dB"
        loss = f", LR loss {r['lr_loss']:.1f}" if r["lr_loss"] is not None else ""
        return f"PSNR {r['psnr']:.2f} dB{loss}\n{bands}"

    fig = plt.figure(figsize=(17, 9.6))
    grid = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, 1.15))
    show(fig.add_subplot(grid[0, 0]), hr, f"Original HR (development)\n{class_name(true_label)}")
    show(fig.add_subplot(grid[1, 0]), recon["bicubic"], "Bicubic upsample\n" + label_of("bicubic"))
    for col, cond in ((1, "wrong"), (2, "oracle")):
        name = class_name(rows[f"CSGM {cond}"]["class_label"])
        what = "wrong label" if cond == "wrong" else "correct label"
        show(fig.add_subplot(grid[0, col]), recon[f"CSGM {cond}"], f"CSGM, {what}: {name}\n" + label_of(f"CSGM {cond}"))
        show(fig.add_subplot(grid[1, col]), recon[f"IADrift {cond}"], f"IADrift, {what}: {name}\n" + label_of(f"IADrift {cond}"))
    ax = fig.add_subplot(grid[:, 3])
    for method, style in zip(METHODS, ("k:", "C0--", "C0-", "C1--", "C1-")):
        f, snr = radial_snr_db(recon[method], hr)
        ax.plot(f, snr, style, label=method)
    ax.axvline(LR_NYQUIST, color="gray", linewidth=1)
    ax.text(LR_NYQUIST, ax.get_ylim()[1], " LR Nyquist", va="top", fontsize=8, color="gray")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylim(-25, 50)  # bicubic is ~90 dB at the lowest bins; keep the rest readable
    ax.set(xlabel="spatial frequency (cycles / HR pixel)", ylabel="SNR (dB): content vs error",
           title="Reconstruction quality by frequency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"Development image {i}  |  SR x4  |  CSGM 5,000 steps  →  IADrift  |  eager evaluation", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.07, top=0.9, wspace=0.2, hspace=0.25)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--images", type=int, nargs="*", help="explicit development image indices")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--lr-theta", type=float, default=1e-5)
    parser.add_argument("--lr-eps", type=float, default=1e-3)
    args = parser.parse_args()

    index_rows = load_index()
    labels = np.array([int(r["label"]) for r in index_rows])
    train_indices, test_indices = load_split(labels)
    images = args.images or pick_dev_images(args.num_images, labels, train_indices, test_indices)
    check_dev(images, train_indices, test_indices)
    lr_all = np.load(DATA_DIR / "imageNet_x4.npy", mmap_mode="r")

    tag = f"steps{args.steps}_lrt{args.lr_theta:g}_lre{args.lr_eps:g}"
    out_dir = DATA_DIR / "iadrift_dev" / tag
    fig_dir = FIGURES_DIR / "iadrift_dev" / tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    results = [json.loads(l) for l in results_path.open()] if results_path.exists() else []
    done = {r["image_index"] for r in results}
    opt_theta, opt_eps, joint_step = make_joint_step(args.lr_theta, args.lr_eps)
    print(f"Development images: {images}\nOutput: {out_dir}", flush=True)

    for i in images:
        if i in done:
            print(f"Image {i}: already done, skipping", flush=True)
            continue
        true_label = int(labels[i])
        hr = load_hr256(DATA_DIR / index_rows[i]["relative_path"])
        y = jnp.asarray(lr_all[i:i + 1], dtype=jnp.float32)
        recon = {"bicubic": bicubic_up(lr_all[i])}
        rows = {"bicubic": {"class_label": None, "lr_loss": None}}
        extra = {}

        for cond, label in (("oracle", true_label), ("wrong", wrong_label_for(i, true_label))):
            restarts = [csgm_cached(i, cond, r, label, lr_all[i], hr) for r in range(FROZEN_CONFIG["num_restarts"])]
            best_r = min(range(len(restarts)), key=lambda r: (restarts[r][0]["final_loss"], r))
            metrics, arrays = restarts[best_r]
            print(f"Image {i} {cond:<6} ({class_name(label)}): CSGM r{best_r} loss {metrics['final_loss']:.2f}, "
                  f"PSNR {metrics['psnr']:.2f} dB; running IADrift...", flush=True)
            recon[f"CSGM {cond}"] = arrays["x_hat"]
            rows[f"CSGM {cond}"] = {"class_label": label, "lr_loss": metrics["final_loss"], "csgm_restart": best_r}

            noise_labels = jnp.asarray(arrays["noise_labels"])
            x, eps, info = run_iadrift(arrays["eps"], noise_labels, label, y, hr,
                                       joint_step, opt_theta, opt_eps, args.steps)
            recon[f"IADrift {cond}"] = x
            rows[f"IADrift {cond}"] = {"class_label": label, "lr_loss": info["eager_loss"],
                                       "theta_relative_change": info["theta_relative_change"],
                                       "seconds": info["seconds"]}
            extra[cond] = info
            buffer = io.BytesIO()
            np.savez_compressed(buffer, x_hat=x, eps=eps, noise_labels=np.asarray(noise_labels),
                                **{k: np.asarray(v) for k, v in info.items() if k.endswith(("_steps", "_loss", "_trace"))})
            atomic_write_bytes(out_dir / f"img{i:05d}_{cond}.npz", buffer.getvalue())
            print(f"Image {i} {cond:<6}: IADrift loss {info['eager_loss']:.2f}, PSNR {info['psnr']:.2f} dB, "
                  f"{info['seconds']:.0f}s", flush=True)

        for method in METHODS:
            rows[method]["psnr"] = psnr(hr, recon[method])
            rows[method]["band_snr_db"] = band_snr_db(recon[method], hr)
        result = {"image_index": i, "true_label": true_label, "steps": args.steps,
                  "lr_theta": args.lr_theta, "lr_eps": args.lr_eps, "methods": rows,
                  "iadrift_psnr_trace": {c: [extra[c]["psnr_steps"], extra[c]["psnr_trace"]] for c in extra}}
        results.append(result)
        atomic_write_bytes(results_path, "".join(json.dumps(r) + "\n" for r in results).encode())
        plot_image(fig_dir / f"img{i:05d}.png", i, true_label, hr, recon, rows)

        print(f"Image {i} band SNR (dB):  " + "  ".join(n for n, *_ in BANDS), flush=True)
        for method in METHODS:
            b = rows[method]["band_snr_db"]
            print(f"  {method:<15} PSNR {rows[method]['psnr']:6.2f}  |  "
                  + "  ".join(f"{b[n]:6.2f}" for n, *_ in BANDS), flush=True)

    # Summary over all finished images in this folder, printed and saved as tables.
    n = len(results)
    print(f"\nMean over {n} development images (band SNR in dB):")
    print(f"{'method':<16}{'PSNR':>8}" + "".join(f"{name:>11}" for name, *_ in BANDS))
    for method in METHODS:
        p = np.mean([r["methods"][method]["psnr"] for r in results])
        b = [np.mean([r["methods"][method]["band_snr_db"][name] for r in results]) for name, *_ in BANDS]
        print(f"{method:<16}{p:>8.2f}" + "".join(f"{v:>11.2f}" for v in b))
    write_tables(results, out_dir, fig_dir)
    print(f"\nFigures and tables: {fig_dir}")


def write_tables(results, out_dir, fig_dir):
    """metrics_per_image.csv and metrics_summary.md (mean over images), in both folders."""
    bands = [name for name, *_ in BANDS]
    header = ["image_index", "class", "method", "label_used", "psnr_db", "lr_loss"] + [f"snr_{b}_db" for b in bands]
    lines = [",".join(header)]
    for r in results:
        for method in METHODS:
            m = r["methods"][method]
            label = class_name(m["class_label"]) if m["class_label"] is not None else ""
            loss = f"{m['lr_loss']:.3f}" if m["lr_loss"] is not None else ""
            lines.append(",".join(
                [str(r["image_index"]), f'"{class_name(r["true_label"])}"', method, f'"{label}"',
                 f"{m['psnr']:.3f}", loss] + [f"{m['band_snr_db'][b]:.3f}" for b in bands]))
    csv_text = "\n".join(lines) + "\n"

    first = results[0]
    md = [f"# IADrift development metrics (SR x4)",
          "",
          f"{len(results)} development images: {', '.join(str(r['image_index']) for r in results)}. "
          f"IADrift: {first['steps']:,} steps, lr theta {first['lr_theta']:g}, lr eps {first['lr_eps']:g}. "
          "CSGM: lr 0.02, 5,000 steps, best of 2 restarts by LR loss.",
          "",
          "Band SNR = 10 log10(HR content / error) per frequency band; higher is better, 0 dB = band left empty, "
          "negative = added detail worse than none. Bands in cycles/HR pixel: low 0-0.0625, mid 0.0625-0.125 "
          "(both inside the LR measurement), high 0.125-0.25, very high > 0.25.",
          "",
          "## Mean over images",
          "",
          "| Method | PSNR (dB) | LR loss | " + " | ".join(f"SNR {b} (dB)" for b in bands) + " |",
          "|---|---:|---:|" + "---:|" * len(bands)]
    for method in METHODS:
        rows = [r["methods"][method] for r in results]
        loss = f"{np.mean([x['lr_loss'] for x in rows]):.2f}" if rows[0]["lr_loss"] is not None else ""
        md.append(f"| {method} | {np.mean([x['psnr'] for x in rows]):.2f} | {loss} | "
                  + " | ".join(f"{np.mean([x['band_snr_db'][b] for x in rows]):.2f}" for b in bands) + " |")
    md += ["", "## Per image", "", "| Image | Method | Label used | PSNR (dB) | LR loss | "
           + " | ".join(f"SNR {b}" for b in bands) + " |", "|---|---|---|---:|---:|" + "---:|" * len(bands)]
    for r in results:
        for method in METHODS:
            m = r["methods"][method]
            label = class_name(m["class_label"]) if m["class_label"] is not None else ""
            loss = f"{m['lr_loss']:.2f}" if m["lr_loss"] is not None else ""
            md.append(f"| {r['image_index']} ({class_name(r['true_label'])}) | {method} | {label} | "
                      f"{m['psnr']:.2f} | {loss} | " + " | ".join(f"{m['band_snr_db'][b]:.2f}" for b in bands) + " |")
    md_text = "\n".join(md) + "\n"

    for folder in (out_dir, fig_dir):
        atomic_write_bytes(folder / "metrics_per_image.csv", csv_text.encode())
        atomic_write_bytes(folder / "metrics_summary.md", md_text.encode())


if __name__ == "__main__":
    main()
