"""Early noise-label swaps vs CSGM on two fresh development images (development; notebook 04, Section 5 uses its CSGM runs).

Check 10b. Every eps step is the frozen CSGM step (CSGM.train_step), so a run in which no swap is accepted is
bit-identical to CSGM.
  csgm   CSGM exactly as frozen: initialize(i, r), restarts 0 and 1, Adam lr 0.02, 5,000 steps (CSGM.reconstruct).
         These two runs per image are also the restart members r0, r1 of the development ensembles (images 9780, 8405).
  early  the same, but after 100 and 250 steps one swap round: all 32 x 63 = 2,016 single-coordinate changes of z are
         screened by their exact LR loss at the current eps; the best 4 and "no swap" each get 100 more CSGM steps
         (continuing the Adam state); the lowest LR loss wins. 5,000 eps steps along the path in total.
Restart rule for both: lowest final LR loss. HR is never used for a decision. Images: the first 2 draws of the
development pick (pick_dev_images(6): numpy seed 2030 over the 100-image pool of the CSGM settings sweep) that no
earlier check used, i.e. 9780 and 8405; correct label.

Ported from scripts/run_z_swap_confirm_dev.py.

Usage:
    python scripts/dev/run_z_swap_confirm_dev.py --dry-run   # plan and cache status; runs and writes nothing (CPU)
    python scripts/dev/run_z_swap_confirm_dev.py --smoke     # GPU: CSGM trace prefix + the first swap round; no writes
    python scripts/dev/run_z_swap_confirm_dev.py             # run what is missing (resumable)

Outputs (IADRIFT_DATA/z_swap_dev_v1/confirm/): img#####_csgm_r{0,1}.npz (CSGM.reconstruct metrics + class_label;
eps, noise_labels, x_hat, trace_steps, trace_loss), img#####_early_r{0,1}.npz (metrics with the swap rounds; x_hat,
eps, noise_labels).
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
from time import perf_counter

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import load_npz, save_npz
from iadrift.csgm import FROZEN_CONFIG

OUT_DIR = paths.DATA_DIR / "z_swap_dev_v1" / "confirm"
DEV_SEED = 2030
DEV_POOL_PATH = paths.DATA_DIR / "phase1_sr4_oracle_moments_dev_v1_6b22675bd557.jsonl"  # the settings-sweep pool
USED = list(data.DEV_IMAGES_4)  # images of the earlier checks
NUM_NEW = 2
SWAP_AT = (100, 250)
N_FINAL = 4
REFIT = 100
TOTAL = FROZEN_CONFIG["num_steps"]
RESTARTS = range(FROZEN_CONFIG["num_restarts"])
SMOKE_IMAGE = 9780


def pick_dev_images(num_images):
    """scripts/run_sr4_iadrift_dev.py pick_dev_images: a seeded draw from the development sweep's image pool."""
    pool = sorted({json.loads(line)["image_index"] for line in DEV_POOL_PATH.open() if line.strip()})
    return [int(i) for i in np.random.default_rng(DEV_SEED).choice(pool, size=num_images, replace=False)]


def new_images(train, test):
    new = [i for i in pick_dev_images(len(USED) + NUM_NEW) if i not in USED][:NUM_NEW]
    assert set(new) <= set(train.tolist()) and not set(new) & set(test.tolist()), "development images only"
    return new


def all_swaps(cur):
    codes = []
    for i in range(cur.size):
        for k in range(64):
            if k != cur[i]:
                c = cur.copy()
                c[i] = k
                codes.append(c)
    return np.stack(codes)


class EarlySwaps:
    """The original run_early with the package's frozen CSGM objects (train_step, optimizer, measurement_loss)."""

    def __init__(self, csgm):
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp, self.csgm = jax, jnp, csgm
        self.loss_jit = jax.jit(csgm.measurement_loss)

    def swap_round(self, eps, state, cur, y, label):
        jnp, loss_jit, train_step = self.jnp, self.loss_jit, self.csgm.train_step
        base = float(loss_jit(eps, y, label, jnp.asarray(cur[None])))
        codes = all_swaps(cur)
        exact = np.array([float(loss_jit(eps, y, label, jnp.asarray(c[None]))) for c in codes])
        res = []
        for code in [cur] + [codes[f] for f in np.argsort(exact)[:N_FINAL]]:
            e2, s2 = eps, state
            for _ in range(REFIT):
                e2, s2, _ = train_step(e2, s2, y, label, jnp.asarray(code[None]))
            res.append((float(loss_jit(e2, y, label, jnp.asarray(code[None]))), code is cur, code, e2, s2))
        best = min(res, key=lambda t: (t[0], not t[1]))
        info = {"accepted": bool(not best[1]), "lr_loss": best[0], "no_swap_loss": res[0][0],
                "n_improving_before_refit": int((exact < base).sum())}
        return best[3], best[4], best[2].copy(), info

    def run(self, i, restart, label, y, hr, stop_after_rounds=None):
        jnp, c = self.jnp, self.csgm
        eps, z = c.initialize(i, restart)
        state = c.optimizer.init(eps)
        cur = np.asarray(z)[0].copy()
        step, rounds = 0, []
        t0 = perf_counter()
        while step < TOTAL:
            nxt = min([s for s in SWAP_AT if s > step] + [TOTAL])
            for _ in range(nxt - step):
                eps, state, _ = c.train_step(eps, state, y, label, jnp.asarray(cur[None]))
            step = nxt
            if step in SWAP_AT:
                eps, state, cur, info = self.swap_round(eps, state, cur, y, label)
                rounds.append({"at_step": step, **info})
                step += REFIT
                if stop_after_rounds is not None and len(rounds) == stop_after_rounds:
                    return None, rounds
        final = float(self.loss_jit(eps, y, label, jnp.asarray(cur[None])))
        x = c.output(eps, label, jnp.asarray(cur[None]))
        from iadrift.metrics import psnr
        m = {"image": i, "restart": restart, "final_loss": final, "psnr": psnr(hr, x), "seconds": perf_counter() - t0,
             "n_accepted": sum(r["accepted"] for r in rounds), "coords_changed": int((cur != np.asarray(z)[0]).sum()),
             "rounds": rounds}
        return m, {"x_hat": x, "eps": np.asarray(eps), "noise_labels": cur[None].astype(np.int32)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; runs and writes nothing")
    parser.add_argument("--smoke", action="store_true", help="GPU: CSGM trace prefix and first swap round; no writes")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train, test = data.load_split(labels)
    new = new_images(train, test)
    lr_all = data.load_measurements()
    print("new development images:", new, flush=True)
    files = [OUT_DIR / f"img{i:05d}_{kind}_r{r}.npz" for i in new for r in RESTARTS for kind in ("csgm", "early")]

    if args.dry_run:
        print("DRY RUN: nothing is run or written.")
        print(f"CSGM and early swaps (at steps {SWAP_AT}, best {N_FINAL} + no swap refit {REFIT} steps), "
              f"{TOTAL:,} eps steps, restarts {list(RESTARTS)}")
        print(f"Cache {OUT_DIR}: {sum(p.exists() for p in files)} / {len(files)} outputs present")
        return

    import jax.numpy as jnp
    from iadrift.csgm import CSGM
    from iadrift.generator import load_generator

    csgm = CSGM(load_generator())
    early = EarlySwaps(csgm)
    if args.smoke:
        i, r = SMOKE_IMAGE, 0
        label, hr = int(labels[i]), data.hr_image(index_rows, i)
        _, a = csgm.reconstruct(i, r, label, lr_all[i], hr, num_steps=100)
        _, saved = load_npz(OUT_DIR / f"img{i:05d}_csgm_r{r}.npz")
        same_trace = bool(np.array_equal(a["trace_loss"], saved["trace_loss"][:len(a["trace_loss"])]))
        print(f"CSGM image {i} r{r}, 100 steps: trace prefix identical: {same_trace}")
        _, rounds = early.run(i, r, label, jnp.asarray(lr_all[i:i + 1], jnp.float32), hr, stop_after_rounds=1)
        saved_round = load_npz(OUT_DIR / f"img{i:05d}_early_r{r}.npz")[0]["rounds"][0]
        same_round = rounds[0] == saved_round
        print(f"early image {i} r{r}, swap round at step {SWAP_AT[0]}: {rounds[0]} vs saved {saved_round}; "
              f"identical: {same_round}")
        ok = same_trace and same_round
        print("SMOKE PASSED" if ok else "SMOKE FAILED")
        raise SystemExit(0 if ok else 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in new:
        label = int(labels[i])
        hr = data.hr_image(index_rows, i)
        y = jnp.asarray(lr_all[i:i + 1], jnp.float32)
        for r in RESTARTS:
            path = OUT_DIR / f"img{i:05d}_csgm_r{r}.npz"
            if not path.exists():
                m, a = csgm.reconstruct(i, r, label, lr_all[i], hr)
                m["class_label"] = label
                save_npz(path, a, m)
            m = load_npz(path)[0]
            print(f"CSGM image {i} r{r}: LR loss {m['final_loss']:.1f}, PSNR {m['psnr']:.2f} dB", flush=True)
            path = OUT_DIR / f"img{i:05d}_early_r{r}.npz"
            if not path.exists():
                m2, a2 = early.run(i, r, label, y, hr)
                save_npz(path, a2, m2)
            m2 = load_npz(path)[0]
            acc = [rr["at_step"] for rr in m2["rounds"] if rr["accepted"]]
            print(f"EARLY image {i} r{r}: swaps accepted at {acc}; LR loss {m2['final_loss']:.1f}, "
                  f"PSNR {m2['psnr']:.2f} dB (CSGM {m['psnr']:.2f}); {m2['seconds']:.0f} s", flush=True)
    print("ALL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
