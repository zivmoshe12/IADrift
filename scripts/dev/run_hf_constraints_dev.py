"""Two high-frequency constraints: a colour penalty in CSGM and a frequency anchor in IADrift (development, image 403;
notebook 04 §6).

Check 16, one development image (403, crane), correct label. HP keeps radial frequencies >= 0.125 cycles/pixel (the LR
limit; circular FFT mask); chroma = (Cb, Cr) of YCbCr. HR is never used for a decision.

A (CSGM stage; default allocator; --a): CSGM with a penalty on high-frequency colour content,
      L(eps) = ||S_s y - A G(eps)||^2 + lam_A ||HP(chroma(G(eps)))||^2,
   otherwise frozen CSGM (Adam lr 0.02 on eps, 5,000 steps). Members as in the development ensemble: restarts r0, r1
   and the r0-start runs on y shifted by (0, 1), (1, 0), (1, 1) LR pixels. lam_A = 0.1 for all members, 0.3 for r0.
B (IADrift stage; platform allocator; --b): IADrift (joint theta + eps, Adam theta lr 1e-5 / eps lr 1e-3, 1,000 steps,
   eager outputs at 100 and 1,000) with an anchor keeping the high band close to the start image x0 (the member's
   CSGM output),
      L(theta, eps) = ||S_s y - A G_theta(eps)||^2 + lam_B ||HP(G_theta(eps) - x0)||^2,   lam_B = 0.02,
   started from every baseline CSGM member ("B") and from every A member with lam_A = 0.1 ("A + B").

The two losses close over the class label (a Python constant) inside the jitted steps, unlike the package's CSGM and
Adaptation steps, so the original code is kept here (with the package's generator calls, which are identical).

Ported from scripts/run_hf_constraints_dev.py.

Usage:
    python scripts/dev/run_hf_constraints_dev.py --dry-run          # plan and cache status; writes nothing (CPU)
    python scripts/dev/run_hf_constraints_dev.py --a --smoke        # GPU: rerun A lam 0.1 shift01 (5,000 steps); exact
    python scripts/dev/run_hf_constraints_dev.py --b --smoke        # GPU: 100 steps of B from base r0; x_step100 exact
    python scripts/dev/run_hf_constraints_dev.py --a                # run the missing A members, then:
    python scripts/dev/run_hf_constraints_dev.py --b                # the missing B members (separate process)

Outputs (IADRIFT_DATA/hf_constraints_dev_v1/): A_lam{0.1,0.3}_<member>.npz (metrics_json {image, member, shift, lam_A,
fit_loss, chroma_hf, seconds}; x_hat, eps, noise_labels), B_from_{base,A}_<member>.npz (metrics_json {image, start,
member, shift, lam_B, step100/step1000: {lr_loss, anchor, finite}, seconds}; x_step100, x_step1000).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

DRY = "--dry-run" in sys.argv
if DRY:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from iadrift.runtime import setup

setup(theta_gradients="--b" in sys.argv, require_gpu=not DRY)  # A: default allocator; B: theta-gradients (platform)

import argparse
from time import perf_counter

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import load_npz, save_npz

I = 403
OUT_DIR = paths.DATA_DIR / "hf_constraints_dev_v1"
CSGM_CACHE = paths.DATA_DIR / "csgm_dev_5000steps_lr0.02"
DEMO_DIR = paths.DATA_DIR / "one_image_demos_v1"
LAM_A = {"r0": (0.1, 0.3), "r1": (0.1,), "shift01": (0.1,), "shift10": (0.1,), "shift11": (0.1,)}
LAM_B = 0.02
SHIFTS = {"r0": (0, 0), "r1": (0, 0), "shift01": (0, 1), "shift10": (1, 0), "shift11": (1, 1)}
STEPS_A, CKPTS_B = 5_000, (100, 1000)
SMOKE_A = ("shift01", 0.1)
SMOKE_B = ("base", "r0")


def a_path(name, lam):
    return OUT_DIR / f"A_lam{lam:g}_{name}.npz"


def b_path(kind, name):
    return OUT_DIR / f"B_from_{kind}_{name}.npz"


def b_starts():
    """{(start kind, member): source file with eps, noise_labels and the start image x_hat}."""
    starts = {}
    for name, (sy, sx) in SHIFTS.items():
        starts[("base", name)] = (CSGM_CACHE / f"img{I:05d}_oracle_r{1 if name == 'r1' else 0}.npz"
                                  if name in ("r0", "r1") else DEMO_DIR / f"D_shift{sy}{sx}_r0.npz")
        starts[("A", name)] = a_path(name, 0.1)
    return starts


def hp_mask(jnp):
    f = jnp.fft.fftfreq(256)
    return (jnp.sqrt(f[:, None] ** 2 + f[None, :] ** 2) >= 0.125)[..., None]


def chroma(jnp, x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return jnp.stack([-0.1687 * r - 0.3313 * g + 0.5 * b, 0.5 * r - 0.4187 * g - 0.0813 * b], axis=-1)


def hp(jnp, x, mask):  # x: [256, 256, C]
    return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(x, axes=(0, 1)) * mask, axes=(0, 1)))


# ---------------------------------------------------------------- A: CSGM + colour penalty
class PartA:
    def __init__(self, gen, label, y0):
        import jax
        import jax.numpy as jnp
        import optax
        from iadrift.operators import A_sr4

        self.jnp, self.gen, self.label, self.y0, self.A_sr4 = jnp, gen, label, y0, A_sr4
        generate = gen.generate
        self.mask = mask = hp_mask(jnp)
        self.opt = opt = optax.adam(0.02)

        def loss(eps, y, z, lam):
            x = generate(eps, label, z).astype(jnp.float32)
            fit = jnp.sum((A_sr4(x) - y) ** 2)
            return fit + lam * jnp.sum(hp(jnp, chroma(jnp, x[0]), mask) ** 2), fit

        @jax.jit
        def step(eps, st, y, z, lam):
            (_, fit), g = jax.value_and_grad(loss, has_aux=True)(eps, y, z, lam)
            u, st = opt.update(g, st, eps)
            return optax.apply_updates(eps, u), st, fit

        self.step = step

    def run(self, name, lam):
        jnp, A_sr4, mask = self.jnp, self.A_sr4, self.mask
        sy, sx = SHIFTS[name]
        r = 1 if name == "r1" else 0
        y = jnp.roll(self.y0, (sy, sx), axis=(1, 2))
        eps, z = self.gen.initialize(I, r)
        st = self.opt.init(eps)
        t0 = perf_counter()
        for _ in range(STEPS_A):
            eps, st, fit = self.step(eps, st, y, z, jnp.float32(lam))
        x = np.asarray(self.gen.generate(eps, self.label, z).astype(jnp.float32))[0]
        fit = float(jnp.sum((A_sr4(jnp.asarray(x)[None]) - y) ** 2))
        meta = {"image": I, "member": name, "shift": [sy, sx], "lam_A": lam, "fit_loss": fit,
                "chroma_hf": float(jnp.sum(hp(jnp, chroma(jnp, jnp.asarray(x)), mask) ** 2)),
                "seconds": perf_counter() - t0}
        return meta, {"x_hat": x, "eps": np.asarray(eps), "noise_labels": np.asarray(z)}


# ---------------------------------------------------------------- B: IADrift + frequency anchor
class PartB:
    def __init__(self, adapt, label, y0):
        import jax
        import jax.numpy as jnp
        import optax
        from iadrift.operators import A_sr4

        self.jax, self.jnp, self.adapt, self.label, self.y0 = jax, jnp, adapt, label, y0
        generate = adapt.gen.generate_theta  # = run_sr4_iadrift_smoke.generate (float32 output)
        self.mask = mask = hp_mask(jnp)
        self.opt_t, self.opt_e = opt_t, opt_e = optax.adam(1e-5), optax.adam(1e-3)

        def loss(theta, eps, y, z, x0):
            x = generate(theta, eps, label, z)
            return jnp.sum((A_sr4(x) - y) ** 2) + LAM_B * jnp.sum(hp(jnp, x[0] - x0, mask) ** 2)

        def _step(theta, eps, st_t, st_e, y, z, x0):
            l, (g_t, g_e) = jax.value_and_grad(loss, argnums=(0, 1))(theta, eps, y, z, x0)
            u, st_t = opt_t.update(g_t, st_t, theta)
            theta = optax.apply_updates(theta, u)
            u, st_e = opt_e.update(g_e, st_e, eps)
            return theta, optax.apply_updates(eps, u), st_t, st_e, l

        self.step = jax.jit(_step, donate_argnums=(0, 1, 2, 3))

    def run(self, kind, name, src, num_steps=CKPTS_B[-1]):
        jnp, mask = self.jnp, self.mask
        sy, sx = SHIFTS[name]
        y = jnp.roll(self.y0, (sy, sx), axis=(1, 2))
        with np.load(src) as d:
            eps, z = jnp.asarray(d["eps"]), jnp.asarray(d["noise_labels"], jnp.int32).reshape(1, -1)
            x0 = jnp.asarray(d["x_hat"])
        theta = self.adapt.fresh_weights()  # = tree_map(jnp.copy, params), as in the original
        st_t, st_e = self.opt_t.init(theta), self.opt_e.init(eps)
        res, meta = {}, {"image": I, "start": kind, "member": name, "shift": [sy, sx], "lam_B": LAM_B}
        t0 = perf_counter()
        for s in range(1, num_steps + 1):
            theta, eps, st_t, st_e, _ = self.step(theta, eps, st_t, st_e, y, z, x0)
            if s in CKPTS_B:
                x, l, _ = self.adapt.eager_metrics(theta, eps, self.label, z, y, np.zeros((256, 256, 3), np.float32))
                res[f"x_step{s}"] = np.asarray(x, np.float32)
                meta[f"step{s}"] = {"lr_loss": l, "anchor": float(jnp.sum(hp(jnp, jnp.asarray(x) - x0, mask) ** 2)),
                                    "finite": bool(np.all(np.isfinite(x)))}
        meta["seconds"] = perf_counter() - t0
        del theta, st_t, st_e
        return meta, res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", action="store_true", help="part A (CSGM + colour penalty; default allocator)")
    parser.add_argument("--b", action="store_true", help="part B (IADrift + frequency anchor; platform allocator)")
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; runs and writes nothing")
    parser.add_argument("--smoke", action="store_true", help="GPU: compare one rerun with the cache; writes nothing")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train, test = data.load_split(labels)
    assert I in set(train.tolist()) and I not in set(test.tolist()), "development images only"
    label = int(labels[I])
    a_files = [a_path(n, lam) for n, lams in LAM_A.items() for lam in lams]
    b_files = [b_path(*key) for key in b_starts()]

    if args.dry_run:
        inputs = [p for p in b_starts().values() if not p.parent == OUT_DIR]
        print("DRY RUN: nothing is run or written.")
        print(f"Image {I}; A: lam_A {LAM_A}, {STEPS_A:,} CSGM steps; B: lam_B {LAM_B}, IADrift outputs at {CKPTS_B}")
        print(f"Cache {OUT_DIR}: A {sum(p.exists() for p in a_files)} / {len(a_files)}, "
              f"B {sum(p.exists() for p in b_files)} / {len(b_files)} outputs present; "
              f"baseline CSGM members (inputs of B) {sum(p.exists() for p in inputs)} / {len(inputs)}")
        return
    if args.a == args.b:
        parser.error("choose exactly one of --a / --b (they need different allocators, so separate processes)")

    import jax.numpy as jnp
    from iadrift.generator import load_generator

    gen = load_generator()
    y0 = jnp.asarray(data.load_measurements()[I:I + 1], jnp.float32)
    if args.a:
        part = PartA(gen, label, y0)
        if args.smoke:
            name, lam = SMOKE_A
            meta, arrays = part.run(name, lam)
            saved_m, saved = load_npz(a_path(name, lam))
            same = meta["fit_loss"] == saved_m["fit_loss"] and all(
                np.array_equal(arrays[k], saved[k]) for k in ("x_hat", "eps", "noise_labels"))
            print(f"A lam {lam:g} {name}: fit LR loss {meta['fit_loss']!r} vs saved {saved_m['fit_loss']!r}; "
                  f"identical: {same}")
            print("SMOKE PASSED" if same else "SMOKE FAILED")
            raise SystemExit(0 if same else 1)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, lams in LAM_A.items():
            for lam in lams:
                path = a_path(name, lam)
                if path.exists():
                    continue
                meta, arrays = part.run(name, lam)
                save_npz(path, arrays, meta)
                print(f"A lam {lam:g} {name}: fit LR loss {meta['fit_loss']:.1f}, chroma HF {meta['chroma_hf']:.0f}, "
                      f"{meta['seconds']:.0f} s", flush=True)
        print("PART A COMPLETE", flush=True)
        return

    from iadrift.adaptation import Adaptation

    part = PartB(Adaptation(gen), label, y0)
    if args.smoke:
        kind, name = SMOKE_B
        meta, res = part.run(kind, name, b_starts()[(kind, name)], num_steps=CKPTS_B[0])
        saved_m, saved = load_npz(b_path(kind, name))
        same = meta["step100"]["lr_loss"] == saved_m["step100"]["lr_loss"] and \
            bool(np.array_equal(res["x_step100"], saved["x_step100"]))
        print(f"B from {kind} {name}, step 100: LR loss {meta['step100']['lr_loss']!r} vs saved "
              f"{saved_m['step100']['lr_loss']!r}; output identical: {same} "
              f"(max |difference| {float(np.max(np.abs(res['x_step100'] - saved['x_step100']))):.2e})")
        print("SMOKE PASSED" if same else "SMOKE FAILED")
        raise SystemExit(0 if same else 1)
    missing_a = [p for p in a_files if not p.exists()]
    if missing_a:
        raise SystemExit(f"Run part A first; missing {[p.name for p in missing_a]}")
    for (kind, name), src in b_starts().items():
        path = b_path(kind, name)
        if path.exists():
            continue
        meta, res = part.run(kind, name, src)
        save_npz(path, res, meta)
        print(f"B from {kind} {name}: LR loss {meta['step100']['lr_loss']:.1f} / {meta['step1000']['lr_loss']:.1f}; "
              f"anchor {meta['step100']['anchor']:.0f} / {meta['step1000']['anchor']:.0f}; {meta['seconds']:.0f} s",
              flush=True)
    print("PART B COMPLETE", flush=True)


if __name__ == "__main__":
    main()
