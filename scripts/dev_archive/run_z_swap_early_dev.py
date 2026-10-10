"""Development images only: discrete noise-label swaps, (2) exhaustive scan at the CSGM solution and
(3) early swaps during a from-scratch CSGM run. z is always a real discrete code; HR is never used for any decision.

(2) SCAN. At each image's best saved CSGM solution, the exact LR loss of all 32 x 63 = 2,016 single-coordinate
    swaps with the current eps (no re-fit), and the first-order (gradient) prediction of each change.
(3) EARLY. From initialize(image, restart), restarts 0 and 1, Adam lr 0.02 on eps, 5,000 eps steps in total.
    After 100, 250, 500 and 1,000 steps, one swap round: all 2,016 single swaps screened exactly with the current
    eps; the best 4 and "no swap" each get 100 eps steps (continuing the Adam state); the lowest LR loss wins.
    Control: the same code without swaps (5,000 steps, same starts, same compiled functions).

Outputs: ~/work/datasets/ImageNetV2/z_swap_dev_v1/{scan,early}/.
Usage:  python scripts/run_z_swap_early_dev.py [--part scan|early|both]
"""

import argparse
import json
from time import perf_counter

from run_z_swap_dev import (
    CSGM_CACHE, DATA_DIR, DEV_IMAGES, FROZEN_CONFIG, NC, NK, OPT, OUT_DIR, check_dev, cond_hard, eps_step,
    gen_cond, grad_W, jax, jnp, load_hr256, load_index, load_split, lr_loss_cond, np, psnr, save,
)
from run_sr4_csgm_test import A_sr4, initialize

SWAP_AT = (100, 250, 500, 1000)
TOTAL_STEPS = FROZEN_CONFIG["num_steps"]  # 5,000
N_FINAL = 4
REFIT = 100


def all_swaps(cur):
    codes, where = [], []
    for i in range(NC):
        for k in range(NK):
            if k != cur[i]:
                c = cur.copy()
                c[i] = k
                codes.append(c)
                where.append((i, k))
    return np.stack(codes), where


@jax.jit
def lr_loss_code(eps, code, label, y):
    # Exact LR loss of a discrete code (cond built inside the compiled function; identical ops to cond_hard).
    return lr_loss_cond(eps, cond_hard(label, code), y)


def exact_losses(eps, codes, label, y):
    return np.array([float(lr_loss_code(eps, jnp.asarray(c[None], jnp.int32), label, y)) for c in codes])


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def scan(i, label, y, hr, out):
    path = out / f"img{i:05d}_scan.npz"
    if path.exists():
        return json.loads(str(np.load(path)["metrics_json"]))
    runs = []
    for r in range(FROZEN_CONFIG["num_restarts"]):
        with np.load(CSGM_CACHE / f"img{i:05d}_oracle_r{r}.npz") as data:
            runs.append((json.loads(str(data["metrics_json"]))["final_loss"], r, data["eps"], data["noise_labels"]))
    _, r_best, eps, z = min(runs, key=lambda t: (t[0], t[1]))
    eps, cur = jnp.asarray(eps), np.asarray(z)[0]
    t0 = perf_counter()
    base = float(lr_loss_code(eps, jnp.asarray(cur[None], jnp.int32), label, y))  # same compiled fn as the swaps
    codes, where = all_swaps(cur)
    exact = exact_losses(eps, codes, label, y)
    g = np.asarray(grad_W(eps, jax.nn.one_hot(jnp.asarray(cur), NK, dtype=jnp.float32), label, y))
    pred = np.array([g[a, k] - g[a, cur[a]] for a, k in where])
    d = exact - base
    m = {"image": i, "csgm_restart": r_best, "base_lr_loss": base, "n_swaps": len(d), "n_improving": int((d < 0).sum()),
         "best_delta": float(d.min()), "median_delta": float(np.median(d)), "spearman_pred_vs_exact": spearman(pred, d),
         "top128_by_pred_best_delta": float(d[np.argsort(pred)[:128]].min()), "seconds": perf_counter() - t0}
    save(path, m, {"exact_delta": d.astype(np.float32), "pred_delta": pred.astype(np.float32),
                   "where": np.asarray(where, np.int16)})
    print(f"SCAN image {i}: base {base:.1f}; improving swaps {m['n_improving']}/{len(d)}; best Δ {m['best_delta']:+.1f}; "
          f"median Δ {m['median_delta']:+.1f}; Spearman(pred, exact) {m['spearman_pred_vs_exact']:+.2f}; {m['seconds']:.0f} s",
          flush=True)
    return m


def early(i, label, y, hr, restart, swaps, out):
    tag = "early" if swaps else "control"
    path = out / f"img{i:05d}_{tag}_r{restart}.npz"
    if path.exists():
        return json.loads(str(np.load(path)["metrics_json"]))
    eps, z = initialize(i, restart)
    state = OPT.init(eps)
    cur = np.asarray(z)[0].copy()
    cond = cond_hard(label, jnp.asarray(cur[None]))
    rounds, extra_steps, extra_forwards, step = [], 0, 0, 0
    t0 = perf_counter()
    while step < TOTAL_STEPS:
        nxt = min([s for s in SWAP_AT if s > step and swaps] + [TOTAL_STEPS])
        for _ in range(nxt - step):
            eps, state = eps_step(eps, state, cond, y)
        step = nxt
        if swaps and step in SWAP_AT:
            base = float(lr_loss_code(eps, jnp.asarray(cur[None], jnp.int32), label, y))  # same compiled fn as the swaps
            codes, where = all_swaps(cur)
            exact = exact_losses(eps, codes, label, y)
            extra_forwards += len(codes)
            branches = [("no swap", cur)] + [(f"z{where[f][0]}:{cur[where[f][0]]}->{where[f][1]}", codes[f])
                                             for f in np.argsort(exact)[:N_FINAL]]
            res = []
            for name, code in branches:
                c = cond_hard(label, jnp.asarray(code[None]))
                e2, s2 = eps, state
                for _ in range(REFIT):
                    e2, s2 = eps_step(e2, s2, c, y)
                res.append((float(lr_loss_cond(e2, c, y)), name, code, e2, s2, c))
            extra_steps += REFIT * (len(branches) - 1)  # the no-swap branch's steps are on the path
            best = min(res, key=lambda t: (t[0], t[1] != "no swap"))
            _, name, code, eps, state, cond = best
            cur = code.copy()
            step += REFIT
            rounds.append({"at_step": step - REFIT, "accepted": None if name == "no swap" else name, "lr_loss": best[0],
                           "no_swap_loss": res[0][0], "screen_base": base, "screen_best_delta": float(exact.min() - base),
                           "screen_n_improving": int((exact < base).sum())})
            print(f"  image {i} r{restart} step {step - REFIT}: {'ACCEPT ' + name if name != 'no swap' else 'no swap'}; "
                  f"LR loss {best[0]:.1f} (no swap {res[0][0]:.1f}); improving swaps before re-fit "
                  f"{rounds[-1]['screen_n_improving']}/{len(codes)}", flush=True)
    x = np.asarray(gen_cond(eps, cond)[0])
    m = {"image": i, "restart": restart, "swaps": swaps, "path_steps": step, "extra_refit_steps": extra_steps,
         "extra_forwards": extra_forwards, "seconds": perf_counter() - t0,
         "lr_loss": float(jnp.sum((A_sr4(jnp.asarray(x)[None]) - y) ** 2)), "psnr": psnr(hr, x),
         "n_accepted": sum(r["accepted"] is not None for r in rounds),
         "coords_changed": int((cur != np.asarray(z)[0]).sum()), "rounds": rounds}
    save(path, m, {"x": x, "eps": np.asarray(eps), "z": cur[None].astype(np.int32), "z0": np.asarray(z)})
    print(f"{tag.upper()} image {i} r{restart}: accepted {m['n_accepted']}, coords changed {m['coords_changed']}; "
          f"LR loss {m['lr_loss']:.1f}, PSNR {m['psnr']:.2f} dB; {m['seconds']:.0f} s", flush=True)
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("scan", "early", "both"), default="both")
    args = parser.parse_args()
    index_rows = load_index()
    labels = np.array([int(r["label"]) for r in index_rows])
    tr, te = load_split(labels)
    check_dev(DEV_IMAGES, tr, te)
    lr_all = np.load(DATA_DIR / "imageNet_x4.npy", mmap_mode="r")
    summary = {}
    for part in (("scan", "early") if args.part == "both" else (args.part,)):
        out = OUT_DIR / part
        out.mkdir(parents=True, exist_ok=True)
        for i in DEV_IMAGES:
            label = int(labels[i])
            hr = load_hr256(DATA_DIR / index_rows[i]["relative_path"])
            y = jnp.asarray(lr_all[i:i + 1], jnp.float32)
            if part == "scan":
                summary[f"scan|{i}"] = scan(i, label, y, hr, out)
            else:
                for r in range(FROZEN_CONFIG["num_restarts"]):
                    summary[f"control|{i}|{r}"] = early(i, label, y, hr, r, False, out)
                    summary[f"early|{i}|{r}"] = early(i, label, y, hr, r, True, out)
        (out / "summary.json").write_text(json.dumps({k: v for k, v in summary.items() if k.startswith(part) or
                                                      (part == "early" and k.startswith("control"))}, indent=1))
    print("ALL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
