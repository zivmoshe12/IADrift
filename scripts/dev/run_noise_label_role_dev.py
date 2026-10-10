"""What do the noise labels z control, and what does eps control? (development, generator only; notebook 04 §1).

No optimization, no HR, no test images: forward passes of the frozen generator (CFG 1.0, eager path) for 8 classes,
the classes of the 4 development images (8117, 3939, 403, 347) plus 4 more drawn with numpy seed 2034.

A. Factorial grid: per class 8 noise-label codes z_k (key fold(fold(PRNGKey(2035), c), k)) x 8 noise inputs eps_j
   (fold(fold(PRNGKey(2036), c), j)), 64 samples cached as float16. For five image descriptors (colour mean/std, pooled
   AlexNet features of the LPIPS network, log radial power spectrum, 16x16 thumbnail, 32x32 edge layout), each
   dimension standardized over the 64 samples, a two-way ANOVA without replication splits the total sum of squares
   into z, eps and interaction shares (summed over dimensions).
B. Locality: from the base sample (z_0, eps_0), 12 single-patch eps resamples and 12 single-coordinate z changes
   (numpy seed [2037, c]); change energy |dx|^2 per 16x16 tile -> share in the perturbed tile / its 3x3 neighbourhood,
   normalized entropy of the tile map (1 = uniform), RMS change.
(Part C of the notebook, the per-tile R3 error maps, only re-reads representation_error_dev_v1/ and caches nothing.)

Ported from notebooks/06_noise_label_role_dev.ipynb (cells 2, 4, 6, 10).

Usage:
    python scripts/dev/run_noise_label_role_dev.py --dry-run   # plan and cache status; runs and writes nothing (CPU)
    python scripts/dev/run_noise_label_role_dev.py --check     # CPU: recompute summary.json from the cache, compare
    python scripts/dev/run_noise_label_role_dev.py --smoke     # GPU: regenerate one grid row, compare bit-exactly
    python scripts/dev/run_noise_label_role_dev.py             # generate what is missing, then summary.json

Outputs (IADRIFT_DATA/noise_label_role_dev_v1/): grid_class###.npz (x: float16 [8 z, 8 eps, 256, 256, 3]),
locality_class###.json (per perturbation: tile, shares, entropy, RMS change, 16x16 tile-energy map), summary.json
(classes, ANOVA shares per class and descriptor, locality means).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

MODE_CPU = any(a in sys.argv for a in ("--dry-run", "--check"))
if MODE_CPU:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from iadrift.runtime import setup

setup(theta_gradients=False, require_gpu=not MODE_CPU)  # forward passes only; default allocator

import argparse
import io
import json

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import atomic_write_bytes
from iadrift.metrics import RADIUS, WINDOW

OUT_DIR = paths.DATA_DIR / "noise_label_role_dev_v1"
CLASS_SEED, Z_SEED, EPS_SEED, PERT_SEED = 2034, 2035, 2036, 2037
NZ, NE, NPERT = 8, 8, 12


def classes_of(labels):
    rng = np.random.default_rng(CLASS_SEED)
    dev_classes = [int(labels[i]) for i in data.DEV_IMAGES_4]
    extra = [int(c) for c in rng.choice(np.setdiff1d(np.arange(1000), dev_classes), size=4, replace=False)]
    return dev_classes + extra


# ---------------------------------------------------------------- generation (GPU)
class Sampler:
    def __init__(self, gen):
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp, self.gen = jax, jnp, gen

    def _key(self, seed, fold):
        key = self.jax.random.PRNGKey(seed)
        for f in fold:
            key = self.jax.random.fold_in(key, f)
        return key

    def eps(self, seed, *fold):
        return self.jax.random.normal(self._key(seed, fold), (1, 256, 256, 3), dtype=self.jnp.float32)

    def z(self, seed, *fold):
        return self.jax.random.randint(self._key(seed, fold), (1, self.gen.noise_coords), 0, self.gen.noise_classes,
                                       dtype=self.jnp.int32)

    def __call__(self, eps, label, z):
        """Frozen generator, CFG 1.0, eager path (same as every saved CSGM output)."""
        return np.asarray(self.jax.device_get(self.gen.generate(eps, label, z)), dtype=np.float32)[0]

    def grid(self, c, rows=range(NZ)):
        x = np.zeros((len(rows), NE, 256, 256, 3), np.float32)
        for a, k in enumerate(rows):
            z = self.z(Z_SEED, c, k)
            for j in range(NE):
                x[a, j] = self(self.eps(EPS_SEED, c, j), c, z)
        return x


def tile_energy(dx):
    e = (dx ** 2).sum(axis=-1)
    return e.reshape(16, 16, 16, 16).sum(axis=(1, 3))  # [tile row, tile col]


def norm_entropy(t):
    p = t.ravel() / t.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(t.size))


def locality(sample, c):
    jnp = sample.jnp
    eps0, z0 = sample.eps(EPS_SEED, c, 0), sample.z(Z_SEED, c, 0)
    x0 = sample(eps0, c, z0)
    rng = np.random.default_rng([PERT_SEED, c])
    out = {"eps_patch": [], "z_coord": []}
    for _ in range(NPERT):
        tr, tc = (int(v) for v in rng.integers(0, 16, size=2))
        new = np.asarray(eps0).copy()
        new[0, 16 * tr:16 * tr + 16, 16 * tc:16 * tc + 16] = rng.standard_normal((16, 16, 3))
        t = tile_energy(sample(jnp.asarray(new), c, z0) - x0)
        nb = t[max(tr - 1, 0):tr + 2, max(tc - 1, 0):tc + 2].sum()
        out["eps_patch"].append({"tile": [tr, tc], "in_tile": float(t[tr, tc] / t.sum()), "in_3x3": float(nb / t.sum()),
                                 "entropy": norm_entropy(t), "rms_change": float(np.sqrt(t.sum() / (256 * 256 * 3))),
                                 "map": t.tolist()})
    for _ in range(NPERT):
        i = int(rng.integers(0, sample.gen.noise_coords))
        z = np.asarray(z0).copy()
        z[0, i] = (z[0, i] + int(rng.integers(1, sample.gen.noise_classes))) % sample.gen.noise_classes  # a new value
        t = tile_energy(sample(eps0, c, jnp.asarray(z)) - x0)
        out["z_coord"].append({"coord": i, "max_tile": float(t.max() / t.sum()), "entropy": norm_entropy(t),
                               "rms_change": float(np.sqrt(t.sum() / (256 * 256 * 3))), "map": t.tolist()})
    return out


# ---------------------------------------------------------------- analysis (CPU)
def descriptors():
    import lpips
    import torch

    torch.set_num_threads(4)
    net = lpips.LPIPS(net="alex", verbose=False).eval()

    def alex_pooled(x):
        t = torch.from_numpy(np.ascontiguousarray(np.clip(x, -1, 1).transpose(2, 0, 1)[None])).float()
        with torch.no_grad():
            feats = net.net.forward(net.scaling_layer(t))
        return np.concatenate([lpips.normalize_tensor(f).mean(dim=(2, 3))[0].numpy() for f in feats])

    def radial_log_power(x, num_bins=32):
        xm = np.asarray(x, np.float64) - np.mean(x, axis=(0, 1))
        p = np.sum(np.abs(np.fft.fft2(xm * WINDOW, axes=(0, 1))) ** 2, axis=-1)
        edges = np.linspace(0.0, 0.5, num_bins + 1)
        return np.log10(np.array([p[(RADIUS >= lo) & (RADIUS < hi)].mean()
                                  for lo, hi in zip(edges[:-1], edges[1:])]) + 1e-12)

    def thumb(x, s):
        return x.reshape(s, 256 // s, s, 256 // s, -1).mean(axis=(1, 3))

    def edge_layout(x):
        g = np.abs(np.diff(x.mean(-1), axis=0))[:, :-1] + np.abs(np.diff(x.mean(-1), axis=1))[:-1, :]  # 255x255
        return thumb(np.pad(g, ((0, 1), (0, 1)))[..., None], 32).ravel()

    return {  # name: (kind, function image -> 1-D vector)
        "colour mean/std": ("position-independent",
                            lambda x: np.concatenate([x.mean(axis=(0, 1)), x.std(axis=(0, 1))])),
        "pooled AlexNet features": ("position-independent", alex_pooled),
        "radial power spectrum": ("position-independent", radial_log_power),
        "16×16 thumbnail (layout)": ("position-specific", lambda x: thumb(x, 16).ravel()),
        "edge layout (32×32 gradient)": ("position-specific", edge_layout),
    }


def anova_shares(D):
    """D: [NZ, NE, d]. Two-way ANOVA without replication per (standardized) dimension, summed over dimensions."""
    D = np.asarray(D, np.float64)
    sd = D.reshape(-1, D.shape[-1]).std(axis=0)
    D = (D - D.reshape(-1, D.shape[-1]).mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    m = D.mean(axis=(0, 1))
    rows, cols = D.mean(axis=1), D.mean(axis=0)
    ss_z = D.shape[1] * np.sum((rows - m) ** 2)
    ss_e = D.shape[0] * np.sum((cols - m) ** 2)
    ss_i = np.sum((D - rows[:, None] - cols[None] + m) ** 2)
    total = ss_z + ss_e + ss_i
    return {"z": ss_z / total, "ε": ss_e / total, "interaction": ss_i / total}


def load_grid(c):
    with np.load(OUT_DIR / f"grid_class{c:03d}.npz") as d:
        return d["x"].astype(np.float32)


def make_summary(classes):
    desc = descriptors()
    shares = {}
    for c in classes:
        grid = load_grid(c)
        for name, (_, fn) in desc.items():
            D = np.stack([np.stack([fn(grid[k, j]) for j in range(NE)]) for k in range(NZ)])
            shares[(c, name)] = anova_shares(D)
    loc = {c: json.loads((OUT_DIR / f"locality_class{c:03d}.json").read_text()) for c in classes}
    ep = [r for c in classes for r in loc[c]["eps_patch"]]
    zc = [r for c in classes for r in loc[c]["z_coord"]]
    return {
        "classes": classes,
        "anova_shares": {f"{c}|{n}": shares[(c, n)] for c in classes for n in desc},
        "locality_means": {"eps_patch_in_tile": float(np.mean([r["in_tile"] for r in ep])),
                           "eps_patch_in_3x3": float(np.mean([r["in_3x3"] for r in ep])),
                           "eps_patch_entropy": float(np.mean([r["entropy"] for r in ep])),
                           "z_coord_entropy": float(np.mean([r["entropy"] for r in zc]))},
    }


def max_abs_diff(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        return max(max_abs_diff(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return max(max_abs_diff(u, v) for u, v in zip(a, b))
    return abs(float(a) - float(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; runs and writes nothing")
    parser.add_argument("--check", action="store_true", help="CPU: recompute summary.json from the cache; no writes")
    parser.add_argument("--smoke", action="store_true", help="GPU: regenerate one grid row and compare; writes nothing")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train, test = data.load_split(labels)
    assert set(data.DEV_IMAGES_4) <= set(train.tolist()) and not set(data.DEV_IMAGES_4) & set(test.tolist())
    classes = classes_of(labels)
    files = ([f"grid_class{c:03d}.npz" for c in classes] + [f"locality_class{c:03d}.json" for c in classes]
             + ["summary.json"])

    if args.dry_run:
        found = [f for f in files if (OUT_DIR / f).exists()]
        print("DRY RUN: nothing is run or written.")
        print(f"Classes: {classes}; grid {NZ} z x {NE} eps per class (seeds z {Z_SEED}, eps {EPS_SEED}); "
              f"{NPERT} + {NPERT} locality perturbations per class (seed {PERT_SEED})")
        print(f"Cache {OUT_DIR}: {len(found)} / {len(files)} outputs present")
        for f in sorted(set(files) - set(found)):
            print("  missing:", f)
        return

    if args.check:
        saved = json.loads((OUT_DIR / "summary.json").read_text())
        fresh = make_summary(classes)
        diff = max_abs_diff(fresh, saved)
        print(f"summary.json recomputed from the cached float16 grids: max |difference| = {diff:.2e}")
        print("CHECK PASSED" if diff < 1e-3 else "CHECK FAILED")
        raise SystemExit(0 if diff < 1e-3 else 1)

    from iadrift.generator import load_generator

    sample = Sampler(load_generator())
    if args.smoke:
        c = 134  # crane (development image 403)
        saved = load_grid(c)[0].astype(np.float16)
        fresh = sample.grid(c, rows=[0])[0].astype(np.float16)
        same = bool(np.array_equal(fresh, saved))
        print(f"grid class {c}, row z0 (8 samples): identical to the cache (float16): {same}")
        print("SMOKE PASSED" if same else "SMOKE FAILED")
        raise SystemExit(0 if same else 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for c in classes:
        path = OUT_DIR / f"grid_class{c:03d}.npz"
        if path.exists():
            continue
        x = sample.grid(c)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, x=x.astype(np.float16))
        atomic_write_bytes(path, buffer.getvalue())
        print(f"saved {path.name}", flush=True)
    for c in classes:
        path = OUT_DIR / f"locality_class{c:03d}.json"
        if not path.exists():
            path.write_text(json.dumps(locality(sample, c)))
            print(f"saved {path.name}", flush=True)
    path = OUT_DIR / "summary.json"
    if not path.exists():  # from the float16 cache, like the saved summary (--check reproduces it exactly)
        atomic_write_bytes(path, json.dumps(make_summary(classes), indent=1).encode())
        print(f"saved {path.name}")
    print("ALL RUNS COMPLETE")


if __name__ == "__main__":
    main()
