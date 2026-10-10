"""Larger posterior-mean ensembles over many noise labels z (development, 6 images).

Extends the 5-member ensembles of notebook 04, Section 5 to K = 12 CSGM members with distinct (eps, z), and tests whether
shorter runs are enough when they are averaged. Correct label, development images only.

Every member is one frozen CSGM run (Adam lr 0.02, CSGM.train_step, ||y - A G(eps; z, c)||^2, 5,000 steps) from
csgm.initialize(i, r), which draws a fresh eps and a fresh z for each r; r0 and r1 are exactly the saved CSGM
restarts, and the runner checks that they match bit for bit. The output, LR loss and eps are saved at steps 500, 1,000,
2,000 and 5,000. Snapshots do not change the trajectory, so one run gives all four step budgets.

Arms (per image, in this order):
    A   distinct z: r = 0..N-1 on y                                          -> img#####_r##.npz
    C   z and grid phase: r = 0..3 on the shifted measurement S_s y = jnp.roll(y, s, axis=(1, 2)) for s in
        iadrift.ensemble.SHIFTS. Each snapshot is shifted back to the original frame (iadrift.ensemble.unshift_x)
        and its LR loss is computed there, as for Ens-W (protocol Section 11)       -> img#####_r##_shift##.npz

IADrift continuation (--iadrift, a separate process: gradients w.r.t. theta need the 'platform' allocator, see
iadrift.runtime). From each arm-A member's CSGM solution at step 5,000, IADrift as frozen (protocol Section 4): joint Adam on
all generator weights theta (lr 1e-5) and on eps (lr 1e-3), fresh optimizer states, 100 steps against y. This is
IADrift @100 of that member                                                  -> img#####_r##_iad100.npz

HR is used only for the PSNR stored in metrics_json (scoring), never to select, weight or stop anything.

Usage (the full run, ~3 h on an RTX A4000; about 1 min per CSGM member, 6 s per IADrift continuation):
    python scripts/dev/run_z_ensemble_dev.py --dry-run                     # plan and cache status; writes nothing
    python scripts/dev/run_z_ensemble_dev.py --smoke                       # GPU: 100 steps of 9780 r0 vs saved restart
    python scripts/dev/run_z_ensemble_dev.py --arms A                      # 12 distinct-z members per image
    python scripts/dev/run_z_ensemble_dev.py --iadrift --smoke             # GPU: 403 r0 vs iadrift_ensemble_dev_v1
    python scripts/dev/run_z_ensemble_dev.py --iadrift                     # IADrift @100 from every arm-A member
    python scripts/dev/run_z_ensemble_dev.py --arms C                      # 4 z x 3 shifts per image
Images 403 and 9780 were run first with --members 16; the analysis uses r0..11 on every image.

Outputs (IADRIFT_DATA/z_ensemble_dev_v1/): one .npz per run with x_hat_step{T}, lr_loss_step{T}, eps_step{T} for
T in SNAPSHOTS, noise_labels, trace_steps, trace_loss (every 50 steps, before the update), metrics_json. Resumable:
existing files are skipped; saves are atomic. Analysis: scripts/analysis/export_z_ensemble_results.py.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `iadrift` imports without install

if "--dry-run" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from iadrift.runtime import setup

# CSGM: default allocator; --iadrift: gradients w.r.t. theta, platform allocator (separate process)
setup(theta_gradients="--iadrift" in sys.argv, require_gpu="--dry-run" not in sys.argv)

import argparse
from time import perf_counter

import numpy as np

from iadrift import data, paths
from iadrift.checkpoint import load_npz, save_npz
from iadrift.ensemble import SHIFTS, unshift_x

OUT_DIR = paths.DATA_DIR / "z_ensemble_dev_v1"
CSGM_CACHE = paths.DATA_DIR / "csgm_dev_5000steps_lr0.02"
CONFIRM_DIR = paths.DATA_DIR / "z_swap_dev_v1" / "confirm"
IAD_DIR = paths.DATA_DIR / "iadrift_ensemble_dev_v1"
IMAGES = list(data.DEV_IMAGES_6)
SNAPSHOTS = (500, 1_000, 2_000, 5_000)
NUM_STEPS = SNAPSHOTS[-1]
TRACE_EVERY = 50
MAX_MEMBERS = 16
ARM_C = tuple(range(4))
SMOKE = (9780, 0, 100)  # image, restart, steps
IAD_STEPS = 100
IAD_SMOKE = (403, 0)  # same CSGM start as iadrift_ensemble_dev_v1/img00403_r0.npz


def run_path(i, r, shift=None):
    if shift is not None:
        return OUT_DIR / f"img{i:05d}_r{r:02d}_shift{shift[0]}{shift[1]}.npz"
    return OUT_DIR / f"img{i:05d}_r{r:02d}.npz"


def restart_path(i, r):
    """The saved CSGM restart r of image i (as in run_ensemble_dev.py)."""
    if i in data.DEV_IMAGES_4:
        return CSGM_CACHE / f"img{i:05d}_oracle_r{r}.npz"
    return CONFIRM_DIR / f"img{i:05d}_csgm_r{r}.npz"


def iad_path(i, r):
    return OUT_DIR / f"img{i:05d}_r{r:02d}_iad{IAD_STEPS}.npz"


def build_plan(images, arms, members):
    """[(image, r, shift or None)] in run order: per image, arm A r = 0..members-1, then arm C."""
    plan = []
    for i in images:
        if "A" in arms:
            plan += [(i, r, None) for r in range(members)]
        if "C" in arms:
            plan += [(i, r, s) for r in ARM_C for s in SHIFTS]
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan and cache status; runs and writes nothing")
    parser.add_argument("--smoke", action="store_true", help="GPU: rerun the start of a saved run; writes nothing")
    parser.add_argument("--images", type=int, nargs="+", default=IMAGES, help=f"subset of {IMAGES} (default: all)")
    parser.add_argument("--arms", default="AC", help="A, C or AC (default AC)")
    parser.add_argument("--members", type=int, default=12, help="arm A members r = 0..N-1 (default 12, max 16)")
    parser.add_argument("--iadrift", action="store_true", help="IADrift @100 from the saved arm-A members")
    args = parser.parse_args()

    index_rows = data.load_index()
    labels = data.labels_of(index_rows)
    train, test = data.load_split(labels)
    assert set(IMAGES) <= set(train.tolist()) and not set(IMAGES) & set(test.tolist()), "development images only"
    assert set(args.images) <= set(IMAGES), f"images must be among the development images {IMAGES}"
    assert set(args.arms) <= set("AC"), "arms: A, C"
    assert 1 <= args.members <= MAX_MEMBERS
    lr_all = data.load_measurements()
    if args.iadrift:
        return iadrift_main(args, index_rows, labels, lr_all)
    plan = build_plan(args.images, args.arms, args.members)

    if args.dry_run:
        print("DRY RUN: nothing is run or written.")
        print(f"Images {args.images}; arms {args.arms}; CSGM lr 0.02, {NUM_STEPS} steps, snapshots {SNAPSHOTS}")
        for i in args.images:
            mine = [p for p in plan if p[0] == i]
            print(f"  image {i}: {sum(run_path(*p).exists() for p in mine)} / {len(mine)} runs present")
        print(f"Saved restarts r0, r1 (bit-exact comparison): "
              f"{sum(restart_path(i, r).exists() for i in args.images for r in (0, 1))} / {2 * len(args.images)} present")
        smoke_ref = restart_path(SMOKE[0], SMOKE[1])
        print(f"Smoke reference {smoke_ref}: {'present' if smoke_ref.exists() else 'MISSING (check skipped)'}")
        print(f"Output: {OUT_DIR}; {sum(run_path(*p).exists() for p in plan)} / {len(plan)} runs present")
        return

    import jax.numpy as jnp
    from iadrift.csgm import CSGM
    from iadrift.generator import load_generator
    from iadrift.metrics import psnr
    from iadrift.operators import lr_loss

    csgm = CSGM(load_generator())

    def member_run(i, r, shift=None, num_steps=NUM_STEPS):
        """One frozen CSGM run with snapshots (the loop of CSGM.reconstruct). Returns (metrics, arrays)."""
        label = int(labels[i])
        y0 = np.asarray(lr_all[i], np.float32)
        y = jnp.asarray(y0[None], jnp.float32)
        if shift is not None:
            y = jnp.roll(y, shift, axis=(1, 2))
        eps, z = csgm.initialize(i, r)
        opt_state = csgm.optimizer.init(eps)
        initial_loss = float(csgm.measurement_loss(eps, y, label, z))
        snaps = [s for s in SNAPSHOTS if s <= num_steps]
        arrays, trace_steps, trace_loss = {}, [], []
        t0 = perf_counter()
        for step in range(1, num_steps + 1):
            eps, opt_state, loss_before = csgm.train_step(eps, opt_state, y, label, z)
            if step % TRACE_EVERY == 0:
                trace_steps.append(step - 1)  # loss_before is evaluated before update `step`
                trace_loss.append(float(loss_before))
            if step in snaps:
                x = csgm.output(eps, label, z)
                loss = float(csgm.measurement_loss(eps, y, label, z))
                if shift is not None:
                    arrays[f"lr_loss_shifted_frame_step{step}"] = np.float32(loss)
                    x = unshift_x(x, *shift)
                    loss = lr_loss(x, y0)  # in the original frame, as for Ens-W
                arrays[f"x_hat_step{step}"] = x
                arrays[f"lr_loss_step{step}"] = np.float32(loss)
                arrays[f"eps_step{step}"] = np.asarray(eps, dtype=np.float32)
        eps.block_until_ready()
        elapsed = perf_counter() - t0
        arrays.update(noise_labels=np.asarray(z), trace_steps=np.asarray(trace_steps, np.int32),
                      trace_loss=np.asarray(trace_loss, np.float32))
        hr = data.hr_image(index_rows, i)
        metrics = {"image": i, "restart": r, "shift": list(shift) if shift else None,
                   "class_label": label, "initial_loss": initial_loss, "elapsed_seconds": elapsed,
                   "lr_loss": {str(s): float(arrays[f"lr_loss_step{s}"]) for s in snaps},
                   "psnr": {str(s): psnr(hr, arrays[f"x_hat_step{s}"]) for s in snaps},  # scoring only
                   "finite": bool(all(np.isfinite(arrays[f"x_hat_step{s}"]).all() for s in snaps)
                                  and np.isfinite(arrays["trace_loss"]).all())}
        return metrics, arrays

    if args.smoke:
        i, r, steps = SMOKE
        _, a = member_run(i, r, num_steps=steps)
        ok = bool(np.isfinite(a["trace_loss"]).all())
        print(f"image {i} r{r}: first {steps} steps, trace {a['trace_loss'].tolist()}; finite: {ok}", flush=True)
        ref = restart_path(i, r)
        if ref.exists():
            _, saved = load_npz(ref)
            n = len(a["trace_loss"])
            same = bool(np.array_equal(a["trace_loss"], saved["trace_loss"][:n]))
            print(f"vs saved {ref.name}: {saved['trace_loss'][:n].tolist()}; bit-exact: {same}", flush=True)
            ok = ok and same
        else:
            print(f"Saved restart {ref} not on this machine; bit-exact check skipped.", flush=True)
        print("SMOKE PASSED" if ok else "SMOKE FAILED", flush=True)
        raise SystemExit(0 if ok else 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [p for p in plan if not run_path(*p).exists()]
    print(f"{len(plan) - len(todo)} / {len(plan)} runs present; {len(todo)} to run", flush=True)
    for n, (i, r, shift) in enumerate(todo, 1):
        path = run_path(i, r, shift)
        m, a = member_run(i, r, shift)
        if shift is None and r in (0, 1) and restart_path(i, r).exists():
            saved_m, saved = load_npz(restart_path(i, r))
            check = {"file": restart_path(i, r).name, "final_loss": saved_m["final_loss"],
                     "final_loss_equal": bool(m["lr_loss"][str(NUM_STEPS)] == np.float32(saved_m["final_loss"])),
                     "x_hat_equal": bool(np.array_equal(a[f"x_hat_step{NUM_STEPS}"], saved["x_hat"])),
                     "trace_equal": bool(np.array_equal(a["trace_loss"], saved["trace_loss"]))}
            m["saved_restart"] = check
            print(f"  r{r} vs saved {check['file']}: final LR loss {m['lr_loss'][str(NUM_STEPS)]:.4f} vs "
                  f"{check['final_loss']:.4f}; equal {check['final_loss_equal']}, x_hat equal {check['x_hat_equal']}, "
                  f"trace equal {check['trace_equal']}", flush=True)
        save_npz(path, a, m)
        losses = " ".join(f"{s}:{m['lr_loss'][str(s)]:.1f}" for s in SNAPSHOTS)
        print(f"[{n}/{len(todo)}] {path.name}: LR loss {losses}; PSNR@5000 {m['psnr'][str(NUM_STEPS)]:.2f}; "
              f"finite {m['finite']} ({m['elapsed_seconds']:.0f} s)", flush=True)
        if not m["finite"]:
            print(f"WARNING: non-finite values in {path.name}", flush=True)
    print(f"Saved {sum(run_path(*p).exists() for p in plan)} / {len(plan)} runs to {OUT_DIR}", flush=True)
    print("ALL RUNS COMPLETE", flush=True)


def iadrift_main(args, index_rows, labels, lr_all):
    """IADrift @100 from every saved arm-A member r = 0..members-1 (theta-gradient process)."""
    from iadrift.adaptation import FROZEN_CONFIG

    plan = [(i, r) for i in args.images for r in range(args.members)]
    if args.dry_run:
        print("DRY RUN: nothing is run or written.")
        print(f"IADrift from the arm-A members: images {args.images}, r = 0..{args.members - 1}; theta lr "
              f"{FROZEN_CONFIG['lr_theta']}, eps lr {FROZEN_CONFIG['lr_eps']}, {IAD_STEPS} steps")
        print(f"CSGM member files (inputs): {sum(run_path(i, r).exists() for i, r in plan)} / {len(plan)} present")
        print(f"Outputs: {sum(iad_path(i, r).exists() for i, r in plan)} / {len(plan)} present")
        return

    import jax.numpy as jnp
    from iadrift.adaptation import Adaptation
    from iadrift.generator import load_generator
    from iadrift.operators import lr_loss

    adapt = Adaptation(load_generator())
    opt_theta, opt_eps, joint_step = adapt.make_joint_step(FROZEN_CONFIG["lr_theta"], FROZEN_CONFIG["lr_eps"])

    def run(i, r):
        _, a = load_npz(run_path(i, r))
        eps = jnp.asarray(a["eps_step5000"])
        z = jnp.asarray(a["noise_labels"], jnp.int32).reshape(1, -1)
        y0 = np.asarray(lr_all[i], np.float32)
        hr = data.hr_image(index_rows, i)  # scoring only
        m, out = adapt.run_iadrift(eps, z, int(labels[i]), jnp.asarray(y0[None]), hr, joint_step, opt_theta, opt_eps,
                                   num_steps=IAD_STEPS, checkpoints=(IAD_STEPS,))
        out[f"lr_loss_step{IAD_STEPS}"] = np.float32(lr_loss(out[f"x_step{IAD_STEPS}"], y0))
        m.update(image=i, restart=r, source=run_path(i, r).name, csgm_lr_loss_step5000=float(a["lr_loss_step5000"]))
        return m, out

    if args.smoke:
        i, r = IAD_SMOKE
        m, out = run(i, r)
        s = m[f"step{IAD_STEPS}"]
        x = out[f"x_step{IAD_STEPS}"]
        print(f"image {i} r{r}: IADrift @{IAD_STEPS} LR loss {s['lr_loss']!r}, PSNR {s['psnr']:.2f}; "
              f"finite {s['finite']}", flush=True)
        ok = s["finite"]
        ref = IAD_DIR / f"img{i:05d}_r{r}.npz"
        if ref.exists():
            saved_m, saved = load_npz(ref)
            same = bool(np.array_equal(x, saved["x_step100"]))
            print(f"vs saved {ref.name}: LR loss {saved_m['step100']['lr_loss']!r}; output identical: {same} "
                  f"(max |difference| {float(np.max(np.abs(x - saved['x_step100']))):.2e})", flush=True)
            ok = ok and same
        else:
            print(f"{ref} not on this machine; bit-exact check skipped.", flush=True)
        print("SMOKE PASSED" if ok else "SMOKE FAILED", flush=True)
        raise SystemExit(0 if ok else 1)

    todo = [(i, r) for i, r in plan if not iad_path(i, r).exists()]
    print(f"IADrift: {len(plan) - len(todo)} / {len(plan)} present; {len(todo)} to run", flush=True)
    for n, (i, r) in enumerate(todo, 1):
        if not run_path(i, r).exists():
            print(f"[{n}/{len(todo)}] {run_path(i, r).name} missing; skipped", flush=True)
            continue
        m, out = run(i, r)
        save_npz(iad_path(i, r), out, m)
        s = m[f"step{IAD_STEPS}"]
        print(f"[{n}/{len(todo)}] {iad_path(i, r).name}: LR loss {m['csgm_lr_loss_step5000']:.1f} -> {s['lr_loss']:.1f}; "
              f"PSNR {s['psnr']:.2f}; finite {s['finite']} ({m['elapsed_seconds']:.0f} s)", flush=True)
        if not s["finite"]:
            print(f"WARNING: non-finite values in {iad_path(i, r).name}", flush=True)
    print(f"Saved {sum(iad_path(i, r).exists() for i, r in plan)} / {len(plan)} IADrift runs", flush=True)
    print("ALL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
