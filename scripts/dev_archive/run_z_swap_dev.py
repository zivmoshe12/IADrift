"""Development images only: gradient-guided discrete swaps of the noise labels z (HotFlip / GCG style).

z stays a real discrete code at all times; the gradient is used only to PROPOSE single-coordinate swaps.
Start: each image's best saved CSGM solution (eps_CSGM, z0), correct label, frozen generator.

One round:
  1. Gradient proposal. Write the noise-label part of cond as rows_i = W_i @ E_i with W = one-hot(z) (32 x 64).
     g = dL/dW at that point. For coordinate i, the first-order change of the LR loss when switching
     z_i -> k is  g[i, k] - g[i, z_i].  Keep the TOP_K most negative per coordinate: 32 x TOP_K candidates.
  2. Exact screening. Evaluate the exact LR loss of every candidate code with the current eps
     (one compiled loss function for all). Keep the best N_REFIT.
  3. Re-fit and accept. eps is co-adapted to its z, so each finalist AND the unchanged code ("no swap")
     get REFIT_STEPS of Adam on eps (lr 0.02, continuing the current Adam state). The branch with the
     lowest LR loss after the re-fit wins (same compiled functions for every branch). Choosing "no swap"
     means the round was plain CSGM steps, so the LR loss never increases.
  Stop after MAX_ROUNDS, or after PATIENCE rounds in a row without an accepted swap.
HR is never used for any decision.

Controls (same start): ε-only CSGM continuation for the same number of ε steps on the accepted path, and for
the total number of ε steps spent (all branches); the cached ε-only run (+2,000) of run_z_opt_dev.py for reference.

Outputs: ~/work/datasets/ImageNetV2/z_swap_dev_v1/ (one .npz per image and control, summary.json).
Usage:  python scripts/run_z_swap_dev.py [--images 8117 ...] [--max-rounds N] [--tag NAME]
"""

import argparse
import io
import json
from time import perf_counter

from run_z_opt_dev import (
    A_sr4, CSGM_CACHE, DATA_DIR, E_NOISE, FROZEN_CONFIG, NC, NK, atomic_write_bytes, check_dev, cond_from_rows,
    cond_hard, gen_cond, jax, jnp, load_hr256, load_index, load_split, np, optax, psnr,
)

OUT_DIR = DATA_DIR / "z_swap_dev_v1"
DEV_IMAGES = [8117, 3939, 403, 347]
TOP_K = 4            # proposals per coordinate (32 x 4 = 128 candidates per round)
N_REFIT = 4          # finalists that get an eps re-fit (plus the "no swap" branch)
REFIT_STEPS = 100
MAX_ROUNDS = 12
PATIENCE = 3
LR_EPS = FROZEN_CONFIG["learning_rate"]
E_BF = E_NOISE.astype(jnp.bfloat16).astype(jnp.float32)
OPT = optax.adam(LR_EPS)


def cond_linear(label, W):
    # W: [32, 64]. Exactly the generator's conditioning when W is one-hot (verified in main).
    rows = jnp.einsum("ik,ikd->id", W, E_BF).astype(jnp.bfloat16)[:, None]
    return cond_from_rows(label, rows)


@jax.jit
def lr_loss_cond(eps, cond, y):
    return jnp.sum((A_sr4(gen_cond(eps, cond)) - y) ** 2)


@jax.jit
def grad_W(eps, W, label, y):
    return jax.grad(lambda w: lr_loss_cond(eps, cond_linear(label, w), y))(W)


def screen_losses(eps, z_batch, label, y):
    # z_batch: [B, 1, 32] int codes -> exact LR loss of each (same eps), one at a time with the single
    # compiled loss function (a batched/vmapped version needs ~7-9 GB of host RAM to compile under WSL).
    return np.array([float(lr_loss_cond(eps, cond_hard(label, z), y)) for z in z_batch])


@jax.jit
def eps_step(eps, state, cond, y):
    g = jax.grad(lambda q: lr_loss_cond(q, cond, y))(eps)
    u, state = OPT.update(g, state, eps)
    return optax.apply_updates(eps, u), state


def refit(eps, state, cond, y):
    # REFIT_STEPS Adam steps on eps (a jitted single step in a Python loop: a fori_loop version needs ~9 GB
    # of host RAM to compile). Returns the new eps, Adam state and the LR loss after the re-fit.
    for _ in range(REFIT_STEPS):
        eps, state = eps_step(eps, state, cond, y)
    return eps, state, lr_loss_cond(eps, cond, y)


def run_image(i, label, y, hr, eps, z, max_rounds):
    state = OPT.init(eps)
    loss = float(lr_loss_cond(eps, cond_hard(label, z), y))
    log = {"start_lr_loss": loss, "rounds": []}
    path_steps = total_steps = 0
    no_accept = 0
    t0 = perf_counter()
    for r in range(1, max_rounds + 1):
        W = jax.nn.one_hot(z[0], NK, dtype=jnp.float32)
        g = np.asarray(grad_W(eps, W, label, y))
        cur = np.asarray(z[0])
        pred = g - g[np.arange(NC), cur][:, None]                          # first-order Δloss for every swap
        pred[np.arange(NC), cur] = np.inf
        cands = [(i_, int(k)) for i_ in range(NC) for k in np.argsort(pred[i_])[:TOP_K]]
        codes = np.repeat(cur[None], len(cands), 0)
        for n, (i_, k) in enumerate(cands):
            codes[n, i_] = k
        exact = screen_losses(eps, jnp.asarray(codes[:, None]), label, y)
        base_exact = float(screen_losses(eps, jnp.asarray(cur[None, None]), label, y)[0])
        pred_d = np.array([pred[i_, k] for i_, k in cands])
        rho = float(np.corrcoef(np.argsort(np.argsort(pred_d)), np.argsort(np.argsort(exact - base_exact)))[0, 1])
        finalists = np.argsort(exact)[:N_REFIT]
        branches = [("no swap", cur)] + [(f"z{cands[f][0]}:{cur[cands[f][0]]}->{cands[f][1]}", codes[f]) for f in finalists]
        results = []
        for name, code in branches:
            e2, s2, l2 = refit(eps, state, cond_hard(label, jnp.asarray(code[None])), y)
            results.append((float(l2), name, code, e2, s2))
        total_steps += REFIT_STEPS * len(branches)
        path_steps += REFIT_STEPS
        best = min(results, key=lambda t: (t[0], t[1] != "no swap"))
        accepted = best[1] != "no swap"
        loss, eps, state, z = best[0], best[3], best[4], jnp.asarray(best[2][None])
        no_accept = 0 if accepted else no_accept + 1
        log["rounds"].append({"round": r, "accepted": best[1] if accepted else None, "lr_loss": loss,
                              "no_swap_loss": results[0][0], "best_screen_delta": float(exact.min() - base_exact),
                              "spearman_pred_vs_exact": rho})
        print(f"  image {i} round {r}: {'ACCEPT ' + best[1] if accepted else 'no swap'}; LR loss {loss:.2f} "
              f"(no-swap branch {results[0][0]:.2f}); first-order vs exact rank corr {rho:+.2f}", flush=True)
        if no_accept >= PATIENCE:
            break
    x = np.asarray(gen_cond(eps, cond_hard(label, z))[0])                   # eager, real code
    log.update({"seconds": perf_counter() - t0, "path_steps": path_steps, "total_eps_steps": total_steps,
                "n_accepted": sum(rr["accepted"] is not None for rr in log["rounds"]),
                "coords_changed": int((np.asarray(z) != np.asarray(log.get("z0", z))).sum()),
                "final_lr_loss_eager": float(jnp.sum((A_sr4(jnp.asarray(x)[None]) - y) ** 2)), "psnr": psnr(hr, x)})
    return log, {"x": x, "eps": np.asarray(eps), "z": np.asarray(z)}


def eps_only(label, y, hr, eps, z, steps):
    # CSGM continuation control: same optimizer, same compiled refit body, z fixed.
    state = OPT.init(eps)
    cond = cond_hard(label, z)
    for _ in range(steps // REFIT_STEPS):
        eps, state, _ = refit(eps, state, cond, y)
    x = np.asarray(gen_cond(eps, cond)[0])
    return {"steps": steps, "lr_loss": float(jnp.sum((A_sr4(jnp.asarray(x)[None]) - y) ** 2)), "psnr": psnr(hr, x)}, \
        {"x": x, "eps": np.asarray(eps)}


def save(path, metrics, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, metrics_json=json.dumps(metrics), **arrays)
    atomic_write_bytes(path, buffer.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, nargs="*", default=DEV_IMAGES)
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    out = OUT_DIR / args.tag if args.tag else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    index_rows = load_index()
    labels = np.array([int(r["label"]) for r in index_rows])
    tr, te = load_split(labels)
    check_dev(args.images, tr, te)
    lr_all = np.load(DATA_DIR / "imageNet_x4.npy", mmap_mode="r")

    summary = {}
    for i in args.images:
        label = int(labels[i])
        hr = load_hr256(DATA_DIR / index_rows[i]["relative_path"])
        y = jnp.asarray(lr_all[i:i + 1], jnp.float32)
        runs = []
        for r in range(FROZEN_CONFIG["num_restarts"]):
            with np.load(CSGM_CACHE / f"img{i:05d}_oracle_r{r}.npz") as data:
                runs.append((json.loads(str(data["metrics_json"]))["final_loss"], r, data["eps"], data["noise_labels"]))
        _, r_best, eps0, z0 = min(runs, key=lambda t: (t[0], t[1]))
        eps0, z0 = jnp.asarray(eps0), jnp.asarray(z0)
        # Check: the linear one-hot conditioning reproduces the generator's conditioning exactly.
        c_lin = cond_linear(label, jax.nn.one_hot(z0[0], NK, dtype=jnp.float32))
        assert bool(jnp.all(c_lin == cond_hard(label, z0))), "linear one-hot cond != generator cond"
        print(f"image {i}: start = CSGM r{r_best}; linear one-hot conditioning check passed", flush=True)

        path = out / f"img{i:05d}_swap.npz"
        if path.exists():
            with np.load(path) as data:
                log = json.loads(str(data["metrics_json"]))
        else:
            log, arr = run_image(i, label, y, hr, eps0, z0, args.max_rounds)
            log["coords_changed"] = int((arr["z"] != np.asarray(z0)).sum())
            log["csgm_restart"] = r_best
            arr["z0"] = np.asarray(z0)
            save(path, log, arr)
        summary[f"{i}|swap"] = log
        for name, steps in (("ctrl_path", log["path_steps"]), ("ctrl_total", log["total_eps_steps"])):
            cpath = out / f"img{i:05d}_{name}_{steps}.npz"
            if cpath.exists():
                with np.load(cpath) as data:
                    m = json.loads(str(data["metrics_json"]))
            else:
                m, arr = eps_only(label, y, hr, eps0, z0, steps)
                save(cpath, m, arr)
            summary[f"{i}|{name}"] = m
        print(f"image {i}: swaps accepted {log['n_accepted']}, coords changed {log['coords_changed']}; "
              f"LR loss {log['start_lr_loss']:.1f} -> {log['final_lr_loss_eager']:.1f}, PSNR {log['psnr']:.2f} dB | "
              f"eps-only {summary[f'{i}|ctrl_path']['steps']} steps: {summary[f'{i}|ctrl_path']['lr_loss']:.1f} / "
              f"{summary[f'{i}|ctrl_path']['psnr']:.2f}; {summary[f'{i}|ctrl_total']['steps']} steps: "
              f"{summary[f'{i}|ctrl_total']['lr_loss']:.1f} / {summary[f'{i}|ctrl_total']['psnr']:.2f}; "
              f"{log['seconds']:.0f} s", flush=True)
    atomic_write_bytes(out / "summary.json", json.dumps(summary, indent=1).encode())
    print("ALL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
