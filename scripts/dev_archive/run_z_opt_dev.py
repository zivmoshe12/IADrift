"""Check 9 (development images only): can the noise labels z be optimized, and is joint (eps, z) worth the compute?

The 32 discrete noise labels enter the generator only through the conditioning vector
    cond = E_class[c] + sum_i E_i[z_i] + 0.02 * cfg_term        (summed in bf16, as upstream)
so z can be optimized through a differentiable relaxation, with the generator frozen:
    soft z      per coordinate i, weights w_i = softmax(logits_i) over its 64 embeddings; the
                coordinate's contribution is w_i @ E_i (a convex combination of real embeddings)
    free cond   cond = cond(z0) + s * delta, delta a free 768-d offset (s = rms of the noise-label sum),
                with a penalty lam * mean(delta^2)

Experiments (correct label; 4 development images; HR never used for any choice):
  Fine-tune from the CSGM solution (best saved restart by LR loss), +2,000 steps, same Adam lr 0.02 on eps:
    eps only                      (compute control: just more CSGM steps)
    soft z only (eps frozen)      (is z optimization possible at all?)
    eps + soft z
    eps + free cond offset
  From scratch, compute-matched with the CSGM baseline (same init, 5,000 steps, 2 restarts, lowest final LR loss):
    eps + soft z from initialize(image, restart)

Soft-z runs also report the rounded code (argmax per coordinate), evaluated with the final eps.
Outputs: ~/work/datasets/ImageNetV2/z_opt_dev_v1/ (one .npz per run + summary.json); analysis in notebook 04, Check 9.

Usage:  python scripts/run_z_opt_dev.py [--dry-run]
"""

import argparse
import io
import json
from time import perf_counter

from run_sr4_csgm_test import (
    A_sr4, DATA_DIR, FROZEN_CONFIG, atomic_write_bytes, generate, initialize, jax, jnp, load_hr256,
    load_index, load_split, model, np, optax, params, psnr,
)
from run_sr4_iadrift_dev import CSGM_CACHE, check_dev

OUT_DIR = DATA_DIR / "z_opt_dev_v1"
DEV_IMAGES = [8117, 3939, 403, 347]
LR_EPS = FROZEN_CONFIG["learning_rate"]  # 0.02, as CSGM
LR_LOGITS = 0.1
LR_DELTA = 0.01
LOGIT_INIT = 10.0     # softmax weight ~0.997 on the starting code
LAM_DELTA = 10.0
FT_STEPS = 2_000
SCRATCH_STEPS = FROZEN_CONFIG["num_steps"]  # 5,000
TRACE_EVERY = 50

NC, NK = model.noise_coords, model.noise_classes  # 32, 64
E_NOISE = jnp.stack([params[f"noise_embeds_{i}"]["embedding"] for i in range(NC)])  # [32, 64, 768] float32
E_CLASS = jnp.asarray(params["Embed_0"]["embedding"])  # [1000, 768] float32
BF = jnp.bfloat16


def _cfg_term():
    def f(m):
        t = m.cfg_norm(m.cfg_embedder(jnp.full((1,), 1.0)))
        return t * 0.02
    return model.apply({"params": params}, method=f)


CFG_TERM = _cfg_term()


def cond_from_rows(label, rows):
    # rows: [32, 1, 768] bf16 noise-label contributions. Same ops and order as c_cfg_noise_to_cond.
    cond = E_CLASS[jnp.asarray(label, jnp.int32).reshape(1)].astype(BF)
    for i in range(NC):
        cond = cond + rows[i]
    return (cond + CFG_TERM).astype(BF)


def cond_hard(label, z):
    return cond_from_rows(label, jnp.stack([E_NOISE[i, z[0, i]][None].astype(BF) for i in range(NC)]))


def cond_soft(label, logits):
    w = jax.nn.softmax(logits, axis=-1)  # [32, 64]
    rows = jnp.einsum("ik,ikd->id", w, E_NOISE.astype(BF).astype(jnp.float32)).astype(BF)[:, None]
    return cond_from_rows(label, rows)


def cond_free(label, z0, delta, scale):
    pre = E_CLASS[jnp.asarray(label, jnp.int32).reshape(1)].astype(BF)
    for i in range(NC):
        pre = pre + E_NOISE[i, z0[0, i]][None].astype(BF)
    return (pre + CFG_TERM + scale * delta).astype(BF)


def gen_cond(eps, cond):
    return model.apply({"params": params}, eps, cond, True, method=model.generate_image).astype(jnp.float32)


def make_step(kind, lr_eps, lr_z, lam=0.0):
    """kind: 'soft' (z = logits) or 'free' (z = delta). Returns (opt, jitted step)."""
    opt = optax.multi_transform({"eps": optax.adam(lr_eps) if lr_eps > 0 else optax.set_to_zero(),
                                 "z": optax.adam(lr_z) if lr_z > 0 else optax.set_to_zero()},
                                {"eps": "eps", "z": "z"})

    def loss_fn(p, y, label, z0, scale):
        cond = cond_soft(label, p["z"]) if kind == "soft" else cond_free(label, z0, p["z"], scale)
        fit = jnp.sum((A_sr4(gen_cond(p["eps"], cond)) - y) ** 2)
        reg = lam * jnp.mean(p["z"] ** 2) if kind == "free" else 0.0
        return fit + reg, fit

    @jax.jit
    def step(p, state, y, label, z0, scale):
        (_, fit), g = jax.value_and_grad(loss_fn, has_aux=True)(p, y, label, z0, scale)
        updates, state = opt.update(g, state, p)
        return optax.apply_updates(p, updates), state, fit

    return opt, step


def run(kind, eps0, z0, label, y, hr, steps, lr_eps, lr_z, lam=0.0):
    scale = jnp.sqrt(jnp.mean(jnp.sum(jnp.stack([E_NOISE[i, z0[0, i]] for i in range(NC)]), 0) ** 2))
    if kind == "soft":
        p = {"eps": eps0, "z": LOGIT_INIT * jax.nn.one_hot(z0[0], NK, dtype=jnp.float32)}
    else:
        p = {"eps": eps0, "z": jnp.zeros((1, 768), jnp.float32)}
    opt, step = make_step(kind, lr_eps, lr_z, lam)
    state = opt.init(p)
    cond = (lambda q: cond_soft(label, q["z"])) if kind == "soft" else (lambda q: cond_free(label, z0, q["z"], scale))
    x_start = np.asarray(gen_cond(p["eps"], cond(p))[0])
    trace_s, trace_l = [], []
    t0 = perf_counter()
    for s in range(1, steps + 1):
        p, state, fit = step(p, state, y, label, z0, scale)
        if s % TRACE_EVERY == 0:
            trace_s.append(s - 1)
            trace_l.append(float(fit))
    jax.block_until_ready(p["eps"])
    seconds = perf_counter() - t0
    x = np.asarray(gen_cond(p["eps"], cond(p))[0])
    out = {"x": x, "x_start": x_start, "eps": np.asarray(p["eps"]), "z_param": np.asarray(p["z"]),
           "trace_steps": np.asarray(trace_s, np.int32), "trace_loss": np.asarray(trace_l, np.float32)}
    m = {"steps": steps, "seconds": seconds, "lr_loss": float(jnp.sum((A_sr4(jnp.asarray(x)[None]) - y) ** 2)),
         "psnr": psnr(hr, x), "start_lr_loss": float(jnp.sum((A_sr4(jnp.asarray(x_start)[None]) - y) ** 2)),
         "start_psnr": psnr(hr, x_start)}
    if kind == "soft":
        w = np.asarray(jax.nn.softmax(p["z"], axis=-1))
        z_round = w.argmax(-1)[None].astype(np.int32)
        x_r = np.asarray(gen_cond(p["eps"], cond_hard(label, jnp.asarray(z_round)))[0])
        ent = -(w * np.log(w + 1e-12)).sum(-1) / np.log(NK)
        out.update({"x_rounded": x_r, "z_rounded": z_round})
        m.update({"rounded_lr_loss": float(jnp.sum((A_sr4(jnp.asarray(x_r)[None]) - y) ** 2)), "rounded_psnr": psnr(hr, x_r),
                  "coords_changed": int((z_round != np.asarray(z0)).sum()), "max_weight_mean": float(w.max(-1).mean()),
                  "weight_entropy_mean": float(ent.mean())})
    else:
        m.update({"delta_rms": float(jnp.sqrt(jnp.mean(p["z"] ** 2)))})
    return m, out


def save(path, metrics, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, metrics_json=json.dumps(metrics), **arrays)
    atomic_write_bytes(path, buffer.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    index_rows = load_index()
    labels = np.array([int(r["label"]) for r in index_rows])
    train_idx, test_idx = load_split(labels)
    check_dev(DEV_IMAGES, train_idx, test_idx)
    lr_all = np.load(DATA_DIR / "imageNet_x4.npy", mmap_mode="r")

    # Reproduction test: the differentiable cond path with a hard code must equal the generator exactly.
    i0 = DEV_IMAGES[0]
    eps_t, z_t = initialize(i0, 0)
    a = np.asarray(gen_cond(eps_t, cond_hard(int(labels[i0]), z_t))[0])
    b = np.asarray(jax.device_get(generate(eps_t, int(labels[i0]), z_t)), np.float32)[0]
    c = np.asarray(gen_cond(eps_t, cond_soft(int(labels[i0]), 1e4 * jax.nn.one_hot(z_t[0], NK)))[0])
    print(f"Reproduction: |hard cond - generator| max {np.abs(a - b).max():.2e}; |one-hot soft - generator| max {np.abs(c - b).max():.2e}",
          flush=True)
    assert np.array_equal(a, b), "hard cond path does not reproduce the generator"

    plan = []
    for i in DEV_IMAGES:
        for name, kind, lr_e, lr_z, lam in (("F0_eps_only", "soft", LR_EPS, 0.0, 0.0), ("F1_z_only", "soft", 0.0, LR_LOGITS, 0.0),
                                            ("F2_eps_softz", "soft", LR_EPS, LR_LOGITS, 0.0),
                                            ("F3_eps_freecond", "free", LR_EPS, LR_DELTA, LAM_DELTA)):
            plan.append((i, name, kind, lr_e, lr_z, lam, None))
        for r in range(FROZEN_CONFIG["num_restarts"]):
            plan.append((i, f"S2_scratch_softz_r{r}", "soft", LR_EPS, LR_LOGITS, 0.0, r))
    print(f"{len(plan)} runs ({len(DEV_IMAGES)} images); fine-tune {FT_STEPS} steps, scratch {SCRATCH_STEPS} steps", flush=True)
    if args.dry_run:
        return

    summary = {}
    for i, name, kind, lr_e, lr_z, lam, restart in plan:
        path = OUT_DIR / f"img{i:05d}_{name}.npz"
        label = int(labels[i])
        hr = load_hr256(DATA_DIR / index_rows[i]["relative_path"])
        y = jnp.asarray(lr_all[i:i + 1], jnp.float32)
        if path.exists():
            with np.load(path) as data:
                summary[f"{i}|{name}"] = json.loads(str(data["metrics_json"]))
            continue
        if restart is None:  # fine-tune from the best saved CSGM restart (lowest final LR loss)
            runs = []
            for r in range(FROZEN_CONFIG["num_restarts"]):
                with np.load(CSGM_CACHE / f"img{i:05d}_oracle_r{r}.npz") as data:
                    runs.append((json.loads(str(data["metrics_json"]))["final_loss"], r, data["eps"], data["noise_labels"]))
            _, r_best, eps0, z0 = min(runs, key=lambda t: (t[0], t[1]))
            eps0, z0, steps = jnp.asarray(eps0), jnp.asarray(z0), FT_STEPS
        else:
            eps0, z0 = initialize(i, restart)
            r_best, steps = restart, SCRATCH_STEPS
        m, out = run(kind, eps0, z0, label, y, hr, steps, lr_e, lr_z, lam)
        m.update({"image": i, "variant": name, "csgm_restart": r_best, "lr_eps": lr_e, "lr_z": lr_z, "lam": lam})
        out["z0"] = np.asarray(z0)
        save(path, m, out)
        summary[f"{i}|{name}"] = m
        extra = (f", rounded {m['rounded_psnr']:.2f} dB / LR {m['rounded_lr_loss']:.1f}, {m['coords_changed']} coords changed"
                 if kind == "soft" else f", delta rms {m['delta_rms']:.3f}")
        print(f"image {i} {name}: LR loss {m['start_lr_loss']:.1f} -> {m['lr_loss']:.1f}; PSNR {m['start_psnr']:.2f} -> "
              f"{m['psnr']:.2f} dB{extra}; {m['seconds']:.0f} s", flush=True)
    atomic_write_bytes(OUT_DIR / "summary.json", json.dumps(summary, indent=1).encode())
    print("ALL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
