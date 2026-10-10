# Test protocol (SR ×4)

We wrote this protocol before running IADrift on the test images and froze it on 2026-10-05. Sections 1–7 are the
plan; Sections 8–11 were added afterwards and are dated. The wording has been tidied for the code release, but no
setting, date or result has changed.

Some file names below are the names the runs had at the time. In this repository they are:

| At the time | Here |
|---|---|
| `run_sr4_csgm_test.py` | `scripts/test/run_csgm_test.py` (and the loop in `iadrift/csgm.py`) |
| `run_sr4_iadrift_test.py`, `run_overnight_iadrift_test.sh` | `scripts/test/run_iadrift_test.py`, `scripts/test/run_overnight.sh iadrift` |
| `run_sr4_ensemble_test.py`, `run_overnight_ensemble_test.sh` | `scripts/test/run_ensemble_test.py`, `scripts/test/run_overnight.sh ensemble` |
| `freq_metrics.blockiness`, `freq_metrics.band_snr_db` | `iadrift.metrics.blockiness`, `iadrift.metrics.band_snr_db` |
| test analysis (working notebooks) | `scripts/analysis/evaluate_test.py` |

The development evidence cited in Section 4.1 comes from `scripts/dev/run_iadrift_dev.py` (the learning-rate grid)
and `scripts/dev/run_theta_only_dev.py`. Directory names such as `phase1_sr4_csgm_label_test_v2_...` are experiment
caches under `IADRIFT_DATA`, outside the repository.

---

## 1. Data

| | |
|---|---|
| Dataset | ImageNet-V2 MatchedFrequency, 10,000 images, split 80/20 per class (`imageNetV2_split_80_20_seed2026.npz`) |
| Test sample | 100 test images from 100 distinct classes, fixed in `sr4_test_sample_100_seed2028.npz`; the sample order is the analysis order |
| HR | EXIF orientation, RGB, bicubic resize of the short side to 256, centre crop, scaled to [-1, 1] |
| Measurement | y = A_sr4(HR), antialiased JAX bicubic 256 → 64 (`imageNet_x4.npy`) |
| Labels | correct class, and a wrong class (one random other class per image, `WRONG_LABEL_SEED = 2029`) |
| Disclosure | test image 1704 was used for the CSGM and IADrift smoke tests before the benchmark; nothing was tuned on it |

The HR image is used only for scoring. No choice here depends on it or on any test result.

## 2. Generator

`hf://pixel_B_sota` (DiT-B, 16×16 pixel patches, 134M parameters, bf16), class-conditional, guidance scale 1.0,
deterministic mode. Calls follow upstream `models/generator.py`: `c_cfg_noise_to_cond(c, 1.0, noise_labels)`, then
`generate_image(ε, cond, deterministic=True)`.

## 3. Methods on the test set

Every method runs with both labels, except bicubic, which has none. The CSGM benchmark and its post-processing were
frozen and run before this protocol; we restate them so that the whole comparison is defined in one place.

| Method | Definition |
|---|---|
| bicubic | `jax.image.resize(y, 256, cubic, antialias=False)` |
| CSGM | frozen generator, Adam lr 0.02 on ε, 5,000 steps, 2 restarts (`initialize(image, restart)`); keep the restart with the lowest final LR loss |
| IADrift @100 | from the CSGM solution (Section 4), joint θ and ε, output after 100 steps |
| IADrift @1000 | the same run, output after 1,000 steps |
| each of the above + BP | back-projection x + Aᵀ(AAᵀ)⁻¹(y − Ax), conjugate gradients, tolerance 1e-6, at most 500 iterations |

Development-only ablations, not part of the test comparison: frequency fusion, projecting ε onto the sphere, label
search, θ-only adaptation, constrained θ adaptation.

## 4. IADrift

| Setting | Value |
|---|---|
| Start | per image and label, the saved CSGM test run with the lowest final LR loss (`phase1_sr4_csgm_label_test_v2_n100_9e24c78a47ab_arrays/`): its ε, its `noise_labels` and the same class |
| θ | the pretrained `pixel_B_sota` weights, a fresh copy per run |
| Objective | ‖y − A G_θ(ε)‖², the LR measurement loss only |
| Optimizers | two Adam optimizers (optax defaults): θ lr 1e-5 on all 241 parameter leaves, ε lr 1e-3; one jitted joint step with donated buffers |
| Steps | 1,000 per run, one deterministic run per image and label, no restarts |
| Outputs | two checkpoints of the same run, step 100 and step 1,000, each through the eager generator path |
| Numerics | `XLA_PYTHON_CLIENT_ALLOCATOR=platform` set before importing JAX (needed for θ gradients under WSL) |
| Monitoring | LR loss of the compiled step every 50 steps; not used for any choice |
| Runs | 100 images × 2 labels = 200 runs, about 2.5 h on an RTX A4000 |

### 4.1 Learning rates and the step-100 checkpoint

We decided this on 2026-10-05, before any IADrift test run. The alternatives were IADrift at 1,000 steps only, or
waiting a day for a constrained variant; we chose plain IADrift with two checkpoints. All the evidence is from
development images.

| Setting | Decision | Development evidence |
|---|---|---|
| θ lr 1e-5 | the largest rate of the grid | Grid θ lr ∈ {1e-6, 3e-6, 1e-5} × ε lr ∈ {0, 1e-3}, 4 images, 1,000 steps. The best mean PSNR gain over CSGM at any step was +0.30 dB at step 50 for θ lr 1e-5, against +0.10 dB (3e-6, step 50) and +0.05 dB (1e-6, steps 100–150). Smaller rates do not avoid the later decline, they only slow it. All earlier IADrift runs also used 1e-5. |
| ε lr 1e-3 | as in all earlier runs | ε hardly matters: ε lr 0 and 1e-3 gave nearly the same results at every θ lr, and θ-only adaptation reproduced the joint results (18.18 against 18.29 dB at θ lr 1e-5). We keep it nonzero so that the method stays the joint (θ, ε) optimization of IAGAN. |
| Checkpoint at step 100 | declared, not tuned | PSNR peaked early and briefly and then fell at every setting: at step 50 for θ lr 1e-5, and at step 50 for both rates in the 2,000-step θ-only runs. The peak step needs the HR image, so it cannot be a stopping rule. We declared step 100 instead: a round number inside the early regime, with some margin beyond the noisy step-50 peak. |
| Checkpoint at step 1,000 | declared | the length of the development runs; IADrift as specified, without early stopping |
| Both from one run | — | no extra compute, and no step count is selected on test data |

We did not choose any step count, learning rate or variant from test images, and we did not run the constrained
variants (weight anchor, frequency anchor, restricted layers) before freezing.

## 5. Metrics

All against the 256×256 HR image; final images from the eager path.

| Metric | Definition |
|---|---|
| PSNR (dB) | data range 2, unclipped; the mean of per-image PSNR, and the PSNR of the mean MSE (IAGAN's convention); clipped PSNR as a secondary number |
| LPIPS | AlexNet backbone, `lpips` 0.1.4, inputs clipped to [-1, 1]; lower is better |
| Blockiness | mean pixel jump across the 16-pixel tile borders divided by the mean jump at the other 4-pixel-aligned positions; about 1 means no tile seams |
| Band SNR | bands low 0–0.0625, mid 0.0625–0.125, high 0.125–0.25, very high > 0.25 cycles/pixel |
| LR loss | ‖y − Ax‖², eager |

## 6. Analysis

Primary comparisons, correct label, metrics PSNR and LPIPS:

1. IADrift @1000 + BP against CSGM + BP
2. IADrift @100 + BP against CSGM + BP

That is 4 primary tests. For each image i let dᵢ = mᵢ(first method) − mᵢ(second method). We report the mean d̄ with
the 95% interval d̄ ± 1.96 s/√n, where s is the sample standard deviation of the dᵢ (divisor n − 1) and n = 100; the
Wilcoxon signed-rank p-value; and the number of images on which the first method is better. The p-values are
Holm-corrected over the 4 tests.

Secondary, uncorrected and exploratory:
- the same comparisons without BP, and with the wrong label;
- IADrift (+ BP) against bicubic + BP;
- the effect of the label on IADrift (correct − wrong), with and without BP;
- blockiness, band SNR and LR loss for all methods;
- visual examples: the first 4 images of the fixed sample (not chosen by result), full image and an 80×80 crop.

## 7. Running and failures

- The runner is resumable (32-run chunks), writes a manifest with a settings hash, locks with `flock`, writes
  atomically, and has `--dry-run` and `--num-images N`. It runs overnight inside `tmux`.
- Outputs: `phase2_sr4_iadrift_test_v1_n100_00ca59364e18.{jsonl,manifest.json}` and
  `…_arrays/img#####_{oracle,wrong}.npz` (outputs and ε at steps 100 and 1,000, the loss trace). Every run also
  regenerates its CSGM start and records the largest difference from the saved CSGM output, as a pipeline check.
- Before the full run: `--dry-run`, then one image (1704, already disclosed) only to check that the pipeline runs.
- Non-finite values or a crash: rerun the identical run. A method failure, such as divergence, is reported as a
  failure and not rerun with other settings.

## 8. Deviations

- **2026-10-05, infrastructure only.** The first 32-run chunk crashed inside the Python interpreter after 16 runs
  (exit 139). The 16 runs had been saved and were kept. We changed the driver to 10-run chunks, counting a chunk as
  failed only if it saved no new run, and resumed from run 17 with the same settings.

## 9. Outcome (added 2026-10-05, after the run)

All 200 runs completed, all outputs finite, and every regenerated CSGM start matched the saved one. The analysis
cache is `iadrift_test_postproc_v1/`, with the primary tests in `primary.json` (in this repository:
`results/primary_iadrift.json`).

Primary results, 100 test images, correct label, Holm over 4 tests, all significant:

| against CSGM + BP | PSNR | LPIPS (lower is better) |
|---|---|---|
| IADrift @1000 + BP | −2.92 dB [−3.18, −2.66], Holm p 1.6e-17, better on 0/100 | +0.082 [+0.066, +0.097], Holm p 1.0e-14, better on 12/100 |
| IADrift @100 + BP | −0.12 dB [−0.20, −0.04], Holm p 1.4e-3, better on 33/100 | −0.008 [−0.013, −0.004], Holm p 3.8e-5, better on 72/100 |

Both checkpoints were declared in Section 4.1, so both are reported. That step 100 is better than step 1,000 is a
result, not a choice. Any later work that treats IADrift @100 as "the" method is a post-hoc decision informed by the
test set, so it has to be labelled as such and developed on development images only.

## 10. Development work after the test run

After Section 9 we ran further experiments on development images only: label optimization and swaps, IADrift from
better starts, post-processing on single images, posterior-mean ensembles of CSGM and IADrift runs, and
high-frequency constraints (colour penalty, frequency anchor). None of them ran on the test set and none changes
Sections 1–9. They use 1–6 images and are exploratory.

## 11. Addendum (2026-10-06): posterior-mean ensemble on the test set

This addendum is post-hoc: it was motivated by development results on 6 images obtained after Section 9. We froze it
on 2026-10-06, before any of the runs below. It does not change Sections 1–10.

- **Images:** the 100 test images of Section 1, correct label only.
- **Members, 5 per image:** the saved CSGM test restarts r0 and r1
  (`phase1_sr4_csgm_label_test_v2_n100_9e24c78a47ab_arrays/img#####_oracle_r{0,1}.npz`), and 3 new CSGM runs with the
  Section 3 settings on y shifted circularly by (0,1), (1,0) and (1,1) LR pixels. Each starts from the r0
  initialization (`initialize(image, 0)`), and its output is shifted back by 4 HR pixels per LR pixel. The
  optimization loop is the CSGM test loop, unchanged.
- **Primary method, Ens-W:** x̂ = BP(Σ wₖ xₖ) with wₖ ∝ exp(−(Lₖ − L_min)/T), where Lₖ = ‖y − A xₖ‖² is member k's
  eager LR loss in the original coordinates and T = max(std(L), 1e-6) (population std); the weights sum to 1.
- **Secondary ensembles:** Ens-U (uniform mean of the 5), E1 (mean of r0 and r1), E2 (mean of r0 and the 3 shifts),
  each + BP.
- **Reference:** CSGM + BP as in Section 3, recomputed and checked against `csgm_test_postproc_v1/`.
- **Primary tests, Holm over 2:** Ens-W + BP against CSGM + BP, in PSNR and in LPIPS, with the statistics of
  Section 6.
- **Secondary, uncorrected:** Ens-U, E1 and E2 + BP against CSGM + BP; every ensemble against bicubic + BP; PSNR of
  the mean MSE; blockiness; band SNR; LR losses; the effective number of members; and the MMSE check, i.e. the
  members' mean MSE over the Ens-U MSE before BP, in dB, against the prediction for independent posterior samples,
  10·log₁₀(2/1.2) = 2.22 dB; the first 4 images as visual examples.
- **Runs:** settings hash `ca2d0ec5e6ff`, 10-run chunks, output `csgm_test_ensemble_v1/`, analysis cache
  `csgm_test_ensemble_postproc_v1/`.
- **Checks before launch (2026-10-06, passed):** `--repro-check` reproduced bit-exactly the saved CSGM test run
  1704/oracle/r0 (final loss 100.2454833984375) and the development shifted run 9780, shift (0,1)
  (107.30801391601562). Default JAX allocator.
- **Failures:** as in Section 7.

### 11.1 Outcome (added 2026-10-07, after the run)

All 300 shifted runs completed, all finite, settings hash unchanged. The recomputed CSGM + BP reference is identical
to `csgm_test_postproc_v1/`. Primary tests: `results/ensemble_primary.json`.

Primary results, 100 test images, correct label, Holm over 2, both significant:

| against CSGM + BP | PSNR | LPIPS (lower is better) |
|---|---|---|
| Ens-W + BP | +1.57 dB [+1.40, +1.74], Holm p 7.8e-18, better on 100/100 | −0.032 [−0.040, −0.024], Holm p 5.7e-10, better on 76/100 |

Secondary: Ens-U + BP gains +2.13 dB and −0.037 LPIPS over CSGM + BP; the MMSE check gives 2.33 dB measured against
2.22 dB predicted.

**Deviation, 2026-10-06, infrastructure only.** The first launch was killed by the Linux OOM killer during run 3,
because a concurrent smoke test used the memory. Runs 1–2 had been saved and were kept (the runs are deterministic),
and the driver was relaunched with the same settings.
