"""Check 11 (development images only, post-hoc): does IADrift @100 on top of a better z-start help?

Pipeline tested:  start (CSGM / noise-label search / early swaps)  ->  IADrift 100 steps  ->  (BP, in the notebook).
IADrift exactly as frozen in docs/PROTOCOL.md (joint theta + eps, Adam, theta lr 1e-5 on all leaves, eps lr 1e-3,
make_joint_step), but only the 100-step checkpoint. Post-hoc: "step 100" was found better than step 1,000 on the
TEST set (protocol Section 9); this check is development-only and is reported as such.

Starts (all cached; for each image and start type the restart with the lowest final LR loss):
  CSGM           saved CSGM (csgm_dev_5000steps_lr0.02/) for 8117, 3939, 403, 347; confirm/csgm for 9780, 8405
  search         Check 7 noise-label search (noise_label_search_dev_v1/final_*) for 8117, 3939, 403, 347
  early          Check 10 early swaps (z_swap_dev_v1/early/) for the first 4; Check 10b (confirm/early) for 9780, 8405
Outputs: ~/work/datasets/ImageNetV2/combo_iadrift100_dev_v1/  (x_start and x_100, eager generator path)
"""

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # theta-gradients; must precede JAX import

import io
import json
from time import perf_counter

from run_sr4_iadrift_smoke import (
    DATA_DIR, atomic_write_bytes, eager_metrics, jax, jnp, load_hr256, load_index, make_joint_step, np, params,
)

OUT = DATA_DIR / "combo_iadrift100_dev_v1"
STEPS = 100
DEV4 = [8117, 3939, 403, 347]
NEW2 = [9780, 8405]
CSGM_CACHE = DATA_DIR / "csgm_dev_5000steps_lr0.02"
NLS = DATA_DIR / "noise_label_search_dev_v1"
SWAP = DATA_DIR / "z_swap_dev_v1"


def load(path):
    with np.load(path) as d:
        return json.loads(str(d["metrics_json"])), {k: d[k] for k in d.files if k != "metrics_json"}


def best_of(paths, loss_key):
    runs = [load(p) for p in paths]
    k = min(range(len(runs)), key=lambda j: (runs[j][0][loss_key], j))
    return runs[k], paths[k].name


def starts(i):
    out = {}
    if i in DEV4:
        (m, a), n = best_of([CSGM_CACHE / f"img{i:05d}_oracle_r{r}.npz" for r in (0, 1)], "final_loss")
        out["CSGM"] = (a["eps"], a["noise_labels"], n)
        s = json.loads((NLS / "summary.json").read_text())[str(i)]["best"]
        _, a = load(NLS / f"final_img{i:05d}_{s}.npz")
        out["search"] = (a["eps"], a["noise_labels"], f"final_img{i:05d}_{s}.npz")
        (m, a), n = best_of([SWAP / "early" / f"img{i:05d}_early_r{r}.npz" for r in (0, 1)], "lr_loss")
        out["early"] = (a["eps"], a["z"], n)
    else:
        (m, a), n = best_of([SWAP / "confirm" / f"img{i:05d}_csgm_r{r}.npz" for r in (0, 1)], "final_loss")
        out["CSGM"] = (a["eps"], a["noise_labels"], n)
        (m, a), n = best_of([SWAP / "confirm" / f"img{i:05d}_early_r{r}.npz" for r in (0, 1)], "final_loss")
        out["early"] = (a["eps"], a["noise_labels"], n)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    rows = load_index()
    lr_all = np.load(DATA_DIR / "imageNet_x4.npy", mmap_mode="r")
    opt_t, opt_e, step = make_joint_step(1e-5, 1e-3)
    for i in DEV4 + NEW2:
        label = int(rows[i]["label"])
        hr = load_hr256(DATA_DIR / rows[i]["relative_path"])
        y = jnp.asarray(lr_all[i:i + 1], jnp.float32)
        for name, (eps, z, src) in starts(i).items():
            path = OUT / f"img{i:05d}_{name}.npz"
            if path.exists():
                continue
            eps, z = jnp.asarray(eps), jnp.asarray(z, jnp.int32).reshape(1, -1)
            x0, l0, p0 = eager_metrics(params, eps, label, z, y, hr)
            theta = jax.tree_util.tree_map(jnp.copy, params)
            st_t, st_e = opt_t.init(theta), opt_e.init(eps)
            t0 = perf_counter()
            for _ in range(STEPS):
                theta, eps, st_t, st_e, _ = step(theta, eps, st_t, st_e, y, label, z)
            x1, l1, p1 = eager_metrics(theta, eps, label, z, y, hr)
            m = {"image": i, "start": name, "source": src, "start_lr_loss": l0, "start_psnr": p0,
                 "iad100_lr_loss": l1, "iad100_psnr": p1, "seconds": perf_counter() - t0}
            b = io.BytesIO()
            np.savez_compressed(b, metrics_json=json.dumps(m), x_start=x0, x_iad100=x1)
            atomic_write_bytes(path, b.getvalue())
            del theta, st_t, st_e
            print(f"image {i} {name}: LR loss {l0:.1f} -> {l1:.1f}; PSNR {p0:.2f} -> {p1:.2f} dB ({src}); {m['seconds']:.0f} s",
                  flush=True)
    print("ALL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
