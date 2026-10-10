"""Noise-label search for CSGM (development, 4 images; notebook 04 §4; selection by LR loss only).

z selects most of the scene (notebook 04 §1), but CSGM keeps the z drawn at initialization. Here each development image
(8117, 3939, 403, 347; correct label) gets 16 candidates: the 2 baseline eps starts initialize(i, r), r in {0, 1}, x
8 noise-label sets per start (k = 0: the baseline's own z; k = 1..7: randint(fold(fold(fold(PRNGKey(2033), i), r), k))).
Every candidate gets a 500-step CSGM screening run; the 2 with the lowest screening LR loss are run to the full 5,000
steps, and the one with the lowest final LR loss is the search result. HR is never used for a choice.

The runs use the frozen CSGM loop (CSGM.reconstruct with start = (eps, z)), so candidate k = 0 at full length is the
saved baseline run: asserted on image 8117, restart 1 before the search starts (in the dry run: on the cached files).

Ported from scripts/run_noise_label_search_dev.py (Check 7).

Usage:
    python scripts/dev/run_noise_label_search_dev.py --dry-run   # plan, cache status, cached-repro check (CPU)
    python scripts/dev/run_noise_label_search_dev.py --smoke     # GPU: 100 steps of a screening run; trace prefix
    python scripts/dev/run_noise_label_search_dev.py             # run what is missing (resumable), then summary.json

Outputs (IADRIFT_DATA/noise_label_search_dev_v1/): screen_img#####_r#_nl#.npz (500 steps, 64 runs),
final_img#####_r#_nl#.npz (5,000 steps, 2 per image); metrics_json {image_index, restart, label_set, steps, class_label,
final_loss, seconds} + eps, noise_labels, x_hat, trace_steps, trace_loss; summary.json {image: screen, top, final,
best}.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

if "--dry-run" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from iadrift.runtime import setup

setup(theta_gradients=False, require_gpu="--dry-run" not in sys.argv)  # default allocator (CSGM numerics)

import argparse
import json

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import atomic_write_bytes, load_npz, save_npz
from iadrift.csgm import FROZEN_CONFIG

OUT_DIR = paths.DATA_DIR / "noise_label_search_dev_v1"
CSGM_CACHE = paths.DATA_DIR / "csgm_dev_5000steps_lr0.02"
IMAGES = list(data.DEV_IMAGES_4)
LABEL_SETS = 8  # per eps start; index 0 = the baseline's own noise labels
SEARCH_SEED = 2033
SCREEN_STEPS = 500
TOP_K = 2
RESTARTS = range(FROZEN_CONFIG["num_restarts"])
FULL_STEPS = FROZEN_CONFIG["num_steps"]
REPRO = (8117, 1)
SMOKE = (403, 0, 5, 100)  # image, restart, label set, steps (screening run of the later search winner)


def screen_path(i, r, k):
    return OUT_DIR / f"screen_img{i:05d}_r{r}_nl{k}.npz"


def final_path(i, r, k):
    return OUT_DIR / f"final_img{i:05d}_r{r}_nl{k}.npz"


def tag(rk):
    return f"r{rk[0]}_nl{rk[1]}"


def repro_exact(m, a):
    i, r = REPRO
    saved_m, saved = load_npz(CSGM_CACHE / f"img{i:05d}_oracle_r{r}.npz")
    return m["final_loss"] == saved_m["final_loss"] and np.array_equal(a["x_hat"], saved["x_hat"]), saved_m


class Search:
    def __init__(self, csgm, index_rows, lr_all):
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp, self.csgm, self.index_rows, self.lr_all = jax, jnp, csgm, index_rows, lr_all

    def noise_labels_for(self, i, r, k):
        if k == 0:
            return self.csgm.initialize(i, r)[1]
        jax, gen = self.jax, self.csgm.gen
        key = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(SEARCH_SEED), i), r), k)
        return jax.random.randint(key, (1, gen.noise_coords), 0, gen.noise_classes, dtype=self.jnp.int32)

    def compute(self, i, r, k, label, steps):
        """One CSGM run from (initialize(i, r) eps, label set k) with the frozen loop. Returns (metrics, arrays)."""
        start = (self.csgm.initialize(i, r)[0], self.noise_labels_for(i, r, k))
        hr = data.hr_image(self.index_rows, i)  # only for the PSNR that reconstruct logs (not saved here)
        m, a = self.csgm.reconstruct(i, r, label, self.lr_all[i], hr, num_steps=steps, start=start)
        metrics = {"image_index": i, "restart": r, "label_set": k, "steps": steps, "class_label": label,
                   "final_loss": m["final_loss"], "seconds": m["elapsed_seconds"]}
        return metrics, a

    def run(self, path, i, r, k, label, steps):
        """Cached run (the original file format)."""
        if path.exists():
            return load_npz(path)
        metrics, arrays = self.compute(i, r, k, label, steps)
        save_npz(path, arrays, metrics)
        return metrics, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan, cache status and cached-repro check")
    parser.add_argument("--smoke", action="store_true", help="GPU: 100 steps of one screening run; writes nothing")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train, test = data.load_split(labels)
    assert set(IMAGES) <= set(train.tolist()) and not set(IMAGES) & set(test.tolist()), "development images only"
    lr_all = data.load_measurements()
    print(f"{len(IMAGES)} images x {len(RESTARTS) * LABEL_SETS} candidates, screening {SCREEN_STEPS} steps, "
          f"top {TOP_K} to {FULL_STEPS} steps")

    if args.dry_run:
        screens = [screen_path(i, r, k) for i in IMAGES for r in RESTARTS for k in range(LABEL_SETS)]
        summary_path = OUT_DIR / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        finals = [final_path(i, *(int(v) for v in t[1:].split("_nl")))
                  for i in IMAGES for t in summary.get(str(i), {}).get("top", [])]
        print("DRY RUN: nothing is run or written.")
        print(f"Cache {OUT_DIR}: screening runs {sum(p.exists() for p in screens)} / {len(screens)}; "
              f"final runs {sum(p.exists() for p in finals)} / {len(IMAGES) * TOP_K}; summary.json "
              f"{'present' if summary else 'missing'}")
        if not summary or not all(p.exists() for p in screens + finals):
            return  # nothing complete to check yet
        # The summary's choices must follow from the cached losses (LR loss only).
        for i in IMAGES:
            s = summary[str(i)]
            screen = {(r, k): load_npz(screen_path(i, r, k))[0]["final_loss"]
                      for r in RESTARTS for k in range(LABEL_SETS)}
            top = sorted(screen, key=lambda rk: (screen[rk], rk))[:TOP_K]
            final = {rk: load_npz(final_path(i, *rk))[0]["final_loss"] for rk in top}
            best = min(final, key=lambda rk: (final[rk], rk))
            assert [tag(rk) for rk in top] == s["top"] and tag(best) == s["best"], i
            assert {tag(rk): v for rk, v in screen.items()} == s["screen"], i
        print("summary.json choices follow from the cached LR losses: True")
        i, r = REPRO
        exact, _ = repro_exact(*load_npz(final_path(i, r, 0)))
        print(f"Cached final_img{i:05d}_r{r}_nl0 equals the saved baseline CSGM run: {exact}")
        return

    from iadrift.csgm import CSGM
    from iadrift.generator import load_generator

    search = Search(CSGM(load_generator()), index_rows, lr_all)
    if args.smoke:
        i, r, k, steps = SMOKE
        _, a = search.compute(i, r, k, int(labels[i]), steps)
        _, saved = load_npz(screen_path(i, r, k))
        n = len(a["trace_loss"])
        same = bool(np.array_equal(a["trace_loss"], saved["trace_loss"][:n]))
        print(f"{screen_path(i, r, k).name}: first {steps} steps, trace {a['trace_loss'].tolist()} vs saved "
              f"{saved['trace_loss'][:n].tolist()}; identical: {same}")
        print("SMOKE PASSED" if same else "SMOKE FAILED")
        raise SystemExit(0 if same else 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Reproduction test: label set 0 at full length must equal the saved baseline run.
    i, r = REPRO
    m, a = search.run(final_path(i, r, 0), i, r, 0, int(labels[i]), FULL_STEPS)
    exact, saved_m = repro_exact(m, a)
    print(f"Reproduction test: {m['final_loss']:.4f} vs saved {saved_m['final_loss']:.4f}, identical: {exact}",
          flush=True)
    assert exact, "search runs would not be comparable with the saved baseline"

    summary = {}
    for i in IMAGES:
        label = int(labels[i])
        screen = {}
        for r in RESTARTS:
            for k in range(LABEL_SETS):
                m, _ = search.run(screen_path(i, r, k), i, r, k, label, SCREEN_STEPS)
                screen[(r, k)] = m["final_loss"]
        top = sorted(screen, key=lambda rk: (screen[rk], rk))[:TOP_K]
        final = {}
        for r, k in top:
            m, _ = search.run(final_path(i, r, k), i, r, k, label, FULL_STEPS)
            final[(r, k)] = m["final_loss"]
        best = min(final, key=lambda rk: (final[rk], rk))
        summary[str(i)] = {"screen": {tag(rk): v for rk, v in screen.items()}, "top": [tag(rk) for rk in top],
                           "final": {tag(rk): v for rk, v in final.items()}, "best": tag(best)}
        print(f"image {i}: screening LR loss {min(screen.values()):.1f}..{max(screen.values()):.1f}; "
              f"top {summary[str(i)]['top']} -> final {[round(v, 1) for v in final.values()]}; "
              f"best {summary[str(i)]['best']}", flush=True)
    summary_path = OUT_DIR / "summary.json"
    if summary_path.exists():
        assert json.loads(summary_path.read_text()) == summary, "cached summary.json differs from the cached runs"
    else:
        atomic_write_bytes(summary_path, json.dumps(summary, indent=1).encode())
    print("ALL RUNS COMPLETE")


if __name__ == "__main__":
    main()
