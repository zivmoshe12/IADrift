"""CSGM with eps projected onto the Gaussian shell (development, 4 images; notebook 04 §3, "eps on the sphere").

Does keeping eps on the shell of a typical Gaussian draw (PULSE-like) help CSGM? After every Adam step:
    global sphere      eps <- eps sqrt(d) / ||eps||,  d = 196,608
    per-patch sphere   every 16x16x3 patch of eps (one generator token, 768 values) rescaled to norm sqrt(768)
The start eps is projected too; Adam's moments are left as they are. Everything else is frozen CSGM: initialize(i, r)
for restarts 0 and 1, Adam lr 0.02, 5,000 steps, loss ||y - A G(eps)||^2; the restart with the lowest final LR loss is
kept (never HR). Baseline: the saved runs without projection (csgm_dev_5000steps_lr0.02/).
Development images 8117, 3939, 403, 347, correct label.

Reproduction test (run first in the notebook): the generic step with the identity projection (CSGM.run) must reproduce
the saved CSGM run of image 8117, restart 1 bit for bit (final loss, eps, output, loss trace).

Ported from notebooks/04_postproc_controls_dev.ipynb cells 37 and 39 (Check 4; iadrift.csgm.CSGM.run is cell 37's
run_csgm without the caching).

Usage:
    python scripts/dev/run_eps_projection_dev.py --dry-run       # plan, cache status, cached-repro check (CPU)
    python scripts/dev/run_eps_projection_dev.py --check         # CPU: recompute summary.json, compare; no writes
    python scripts/dev/run_eps_projection_dev.py --smoke         # GPU: 100 steps of one saved run; trace prefix
    python scripts/dev/run_eps_projection_dev.py --repro-check   # GPU: full reproduction test; writes nothing
    python scripts/dev/run_eps_projection_dev.py                 # run what is missing, then summary.json

Outputs (IADRIFT_DATA/csgm_eps_projection_dev_v1/): img#####_{global,perpatch}_sphere_r{0,1}.npz (metrics_json
{final_loss, seconds}; eps, noise_labels, x_hat, trace_steps, trace_loss), repro_img08117_none_r1.npz (the
reproduction run), summary.json (per variant and image: kept restart, metrics raw and + BP, eps statistics).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

MODE_CPU = any(a in sys.argv for a in ("--dry-run", "--check"))
if MODE_CPU:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from iadrift.runtime import setup

setup(theta_gradients=False, require_gpu=not MODE_CPU)  # default allocator: needed for the bit-exact reproduction

import argparse
import json

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import atomic_write_bytes, load_npz, save_npz

OUT_DIR = paths.DATA_DIR / "csgm_eps_projection_dev_v1"
CSGM_CACHE = paths.DATA_DIR / "csgm_dev_5000steps_lr0.02"
IMAGES = list(data.DEV_IMAGES_4)
RESTARTS = (0, 1)
VARIANTS = {"global sphere": "global_sphere", "per-patch sphere": "perpatch_sphere"}  # name: file slug
REPRO = (8117, 1)
SMOKE = (403, "global_sphere", 0, 100)  # image, slug, restart, steps


def run_path(i, slug, r):
    return OUT_DIR / f"img{i:05d}_{slug}_r{r}.npz"


def repro_path():
    return OUT_DIR / f"repro_img{REPRO[0]:05d}_none_r{REPRO[1]}.npz"


def same_run(m, a, saved_m, saved):
    return (m["final_loss"] == saved_m["final_loss"] and all(
        np.array_equal(a[k], saved[k]) for k in ("eps", "x_hat", "trace_loss")))


def check_cached_repro():
    """The cached reproduction run must equal the saved baseline run bit for bit (CPU, reads only)."""
    i, r = REPRO
    m, a = load_npz(repro_path())
    saved_m, saved = load_npz(CSGM_CACHE / f"img{i:05d}_oracle_r{r}.npz")
    return same_run(m, a, saved_m, saved)


# ---------------------------------------------------------------- summary (notebook cell 32 eps_stats, cell 39)
GAUSS_PATCH_STD = 1 / np.sqrt(2 * 768)


def patch_view(e):
    """[.., 256, 256, 3] -> [256 patches, 768], in the generator's patch order."""
    e = np.asarray(e, np.float64).reshape(16, 16, 16, 16, 3)
    return e.transpose(0, 2, 1, 3, 4).reshape(256, 768)


def eps_stats(e, e0=None):
    p = patch_view(e)
    pn = np.linalg.norm(p, axis=1) / np.sqrt(768)
    s = {"rms": float(np.sqrt(np.mean(p ** 2))), "mean": float(p.mean()),
         "patch_norm_mean": float(pn.mean()), "patch_norm_std": float(pn.std()),
         "patch_norm_min": float(pn.min()), "patch_norm_max": float(pn.max()),
         "patch_mean_abs": float(np.abs(p.mean(1)).mean())}
    if e0 is not None:
        s["rel_change"] = float(np.linalg.norm(p - patch_view(e0)) / np.linalg.norm(patch_view(e0)))
    return s


def make_summary(index_rows, lr_all):
    """summary.json as written by notebook 04 cell 39 (metrics of notebook 04 cell 8: PSNR, clipped PSNR, LPIPS,
    LR loss, band SNR; raw and + BP). Needs only the cached runs (CPU)."""
    import jax
    import jax.numpy as jnp
    import torch
    from iadrift.metrics import band_snr_db, lpips_dist, psnr
    from iadrift.operators import backproject

    torch.set_num_threads(4)

    def initial_eps(i, r):  # = generator.initialize(i, r)[0], without loading the model
        key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(data.SWEEP_SEED), i), r)
        return np.asarray(jax.random.normal(jax.random.split(key)[0], (1, 256, 256, 3), dtype=jnp.float32))

    def lr_loss(x, y):  # y: [1, 64, 64, 3], as in the notebook
        from iadrift.operators import A_sr4
        return float(jnp.sum((A_sr4(jnp.asarray(x, jnp.float32)[None]) - y) ** 2))

    def evaluate(x, hr, y):
        return {"psnr": psnr(hr, x), "psnr_clipped": psnr(hr, np.clip(x, -1, 1)), "lpips": lpips_dist(x, hr),
                "lr_loss": lr_loss(x, y), "band_snr_db": band_snr_db(x, hr)}

    def best(paths_by_restart):
        runs = {r: load_npz(p) for r, p in paths_by_restart.items()}
        r = min(runs, key=lambda k: (runs[k][0]["final_loss"], k))
        return r, runs[r][1]

    summary = {"none (saved)": {}, **{v: {} for v in VARIANTS}}
    for i in IMAGES:
        hr = data.hr_image(index_rows, i)
        y = jnp.asarray(lr_all[i:i + 1], jnp.float32)
        kept = {"none (saved)": best({r: CSGM_CACHE / f"img{i:05d}_oracle_r{r}.npz" for r in RESTARTS})}
        for v, slug in VARIANTS.items():
            kept[v] = best({r: run_path(i, slug, r) for r in RESTARTS})
        for v, (r, a) in kept.items():
            x = a["x_hat"]
            x_bp = backproject(x, np.asarray(y[0]))
            summary[v][str(i)] = {"restart": r, "metrics": evaluate(x, hr, y), "metrics_bp": evaluate(x_bp, hr, y),
                                  "eps_stats": eps_stats(a["eps"], initial_eps(i, r))}
    return summary


def max_abs_diff(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        return max(max_abs_diff(a[k], b[k]) for k in a)
    return abs(float(a) - float(b))


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan, cache status and cached-repro check")
    parser.add_argument("--smoke", action="store_true", help="GPU: 100 steps of one saved run; writes nothing")
    parser.add_argument("--repro-check", action="store_true", help="GPU: identity-projection rerun; no writes")
    parser.add_argument("--check", action="store_true", help="CPU: recompute summary.json and compare; writes nothing")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train, test = data.load_split(labels)
    assert set(IMAGES) <= set(train.tolist()) and not set(IMAGES) & set(test.tolist()), "development images only"
    lr_all = data.load_measurements()
    plan = [(i, slug, r) for slug in VARIANTS.values() for i in IMAGES for r in RESTARTS]  # notebook order

    if args.dry_run:
        files = [run_path(*p) for p in plan] + [repro_path(), OUT_DIR / "summary.json"]
        found = sum(p.exists() for p in files)
        print("DRY RUN: nothing is run or written.")
        print(f"Images {IMAGES}; variants {list(VARIANTS)}; restarts {list(RESTARTS)}; CSGM lr 0.02, 5,000 steps")
        print(f"Cache {OUT_DIR}: {found} / {len(files)} outputs present")
        if repro_path().exists():
            print(f"Cached reproduction run equals the saved baseline (image {REPRO[0]}, r{REPRO[1]}): "
                  f"{check_cached_repro()}")
        return
    if args.check:  # the saved summary was computed with JAX on the GPU (BP, LR loss): expect float-level differences
        saved, fresh = json.loads((OUT_DIR / "summary.json").read_text()), make_summary(index_rows, lr_all)
        restarts_same = all(saved[v][k]["restart"] == fresh[v][k]["restart"] for v in saved for k in saved[v])
        worst = {key: max(max_abs_diff(fresh[v][k][key], saved[v][k][key]) for v in saved for k in saved[v])
                 for key in ("metrics", "metrics_bp", "eps_stats")}
        print(f"kept restarts identical: {restarts_same}; max |difference| per block: "
              + ", ".join(f"{k} {d:.2e}" for k, d in worst.items()))
        ok = restarts_same and worst["eps_stats"] < 1e-9 and max(worst.values()) < 1e-3
        print("CHECK PASSED" if ok else "CHECK FAILED")
        raise SystemExit(0 if ok else 1)

    import jax.numpy as jnp
    from iadrift.csgm import CSGM, best_restart, project_global_sphere, project_none, project_patch_sphere
    from iadrift.generator import load_generator

    csgm = CSGM(load_generator())
    projections = {"global_sphere": project_global_sphere, "perpatch_sphere": project_patch_sphere}

    def y_of(i):
        return jnp.asarray(lr_all[i:i + 1], jnp.float32)

    if args.smoke:
        i, slug, r, steps = SMOKE
        _, a = csgm.run(csgm.measurement_loss, projections[slug], y_of(i), int(labels[i]), *csgm.initialize(i, r),
                        num_steps=steps)
        saved_m, saved = load_npz(run_path(i, slug, r))
        k = len(a["trace_loss"])
        same = bool(np.array_equal(a["trace_loss"], saved["trace_loss"][:k]))
        print(f"{run_path(i, slug, r).name}: first {steps} steps, trace {a['trace_loss'].tolist()} vs saved "
              f"{saved['trace_loss'][:k].tolist()}; identical: {same}")
        print("SMOKE PASSED" if same else "SMOKE FAILED")
        raise SystemExit(0 if same else 1)

    i, r = REPRO
    if args.repro_check or not repro_path().exists():
        m, a = csgm.run(csgm.measurement_loss, project_none, y_of(i), int(labels[i]), *csgm.initialize(i, r))
        saved_m, saved = load_npz(CSGM_CACHE / f"img{i:05d}_oracle_r{r}.npz")
        exact = same_run(m, a, saved_m, saved)
        print(f"Reproduction test (image {i}, restart {r}): final LR loss {m['final_loss']:.4f} vs saved "
              f"{saved_m['final_loss']:.4f}; eps, output and loss trace identical: {exact}")
        if args.repro_check:
            print("REPRO CHECK PASSED" if exact else "REPRO CHECK FAILED")
            raise SystemExit(0 if exact else 1)
        assert exact, "The generic CSGM step does not reproduce the saved run; comparisons would be confounded."
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        save_npz(repro_path(), a, m)
    else:
        assert check_cached_repro(), "cached reproduction run differs from the saved baseline"

    for i, slug, r in plan:
        path = run_path(i, slug, r)
        if path.exists():
            continue
        m, a = csgm.run(csgm.measurement_loss, projections[slug], y_of(i), int(labels[i]), *csgm.initialize(i, r))
        save_npz(path, a, m)
        print(f"{slug}, image {i} r{r}: final LR loss {m['final_loss']:.1f} ({m['seconds']:.0f} s)", flush=True)
    for slug in VARIANTS.values():
        for i in IMAGES:
            runs = {r: load_npz(run_path(i, slug, r)) for r in RESTARTS}
            r, *_ = best_restart(runs)
            losses = ", ".join(f"r{k} {runs[k][0]['final_loss']:.1f}" for k in RESTARTS)
            print(f"{slug}, image {i}: final LR loss {losses} -> keep r{r}")
    if not (OUT_DIR / "summary.json").exists():
        atomic_write_bytes(OUT_DIR / "summary.json", json.dumps(make_summary(index_rows, lr_all), indent=1).encode())
        print(f"saved {OUT_DIR / 'summary.json'}")
    print("ALL RUNS COMPLETE")


if __name__ == "__main__":
    main()
