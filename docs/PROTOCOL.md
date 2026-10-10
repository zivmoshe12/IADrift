> **Note for readers of the repository.** This is the project's working protocol, unchanged except for this note, the removal of a pointer to an internal working file, and the description of the paired-test interval in Sections 6 and 11, which now states the interval the code computes. It refers to files by their internal names: `notebooks/03_csgm_test_analysis.ipynb` and `notebooks/07_ensemble_test.ipynb` are in `notebooks/dev_archive/` (outputs cleared); their results are presented in `notebooks/01_sr4_csgm_vs_iadrift.ipynb` and `notebooks/03_ensemble_posterior_mean.ipynb`. Paths under `~/work/datasets/ImageNetV2/` are experiment caches outside the repository.

# IADrift test protocol (SR ×4)

**Status: FROZEN on 2026-10-05, before any IADrift run on test images.**
Choices below were made on development images only (notebooks 02, 04, 05) and confirmed by the project owner on 2026-10-05.
Any later change must be logged in "Deviations" at the end, with the date and reason, and must not be motivated by IADrift test results.

The CSGM test benchmark and its post-processing and metrics were frozen and run earlier; they're restated in Section 3 so the whole test comparison is defined in one place.

---

## 1. Data

| | |
|---|---|
| Dataset | ImageNet-V2 MatchedFrequency, 10,000 images; 80/20 split per class (`imageNetV2_split_80_20_seed2026.npz`) |
| Test sample | **100 test images, 100 distinct classes**, fixed in `sr4_test_sample_100_seed2028.npz` (sample order is the analysis order) |
| HR | EXIF orientation → RGB → bicubic short-side resize to 256 → centre crop → [-1, 1] |
| Measurement | y = A_sr4(HR): antialiased JAX bicubic 256 → 64 (`imageNet_x4.npy`) |
| Label conditions | **correct** (oracle class) and **wrong** (one random other class per image, `WRONG_LABEL_SEED = 2029`) |
| Disclosure | test image **1704** was used for CSGM and IADrift smoke tests before the benchmark; nothing was tuned on it |

HR is used **only** for scoring. No choice in this protocol depends on HR or on any test result.

## 2. Generator

`hf://pixel_B_sota` (DiT-B, 16×16 pixel patches, 134M parameters, bf16 compute), class-conditional, **CFG 1.0**, deterministic mode. Calls as in upstream `models/generator.py`: `c_cfg_noise_to_cond(c, 1.0, noise_labels)` → `generate_image(ε, cond, deterministic=True)`.

## 3. Methods compared on the test set

All methods are evaluated for both label conditions, except bicubic (label-free).

| Method | Definition | Status |
|---|---|---|
| bicubic | `jax.image.resize(y, 256, cubic, antialias=False)` | done |
| CSGM | frozen generator; Adam lr 0.02 on ε; 5,000 steps; 2 restarts (`initialize(image, restart)`), keep the **lowest final LR loss** | done |
| **IADrift @100** | from the CSGM solution (Section 4); joint θ + ε; output after **100** steps | done (2026-10-05; Section 9) |
| **IADrift @1000** | same run; output after **1,000** steps | done (2026-10-05; Section 9) |
| *each of the above* **+ BP** | back-projection x + Aᵀ(AAᵀ)⁻¹(y − A x), conjugate gradients, tolerance 1e-6, at most 500 iterations | done (for IADrift, in its test analysis) |

Not part of the test comparison (development-only ablations): frequency fusion, ε sphere projection, noise-label search, θ-only adaptation, constrained θ adaptation.

## 4. IADrift (frozen)

| Setting | Value |
|---|---|
| Start point | per image and label condition, the **saved CSGM test run with the lowest final LR loss** (`phase1_sr4_csgm_label_test_v2_n100_9e24c78a47ab_arrays/`): its ε and `noise_labels`, the same class label |
| θ initialization | pretrained `pixel_B_sota` weights (a fresh copy per run) |
| Objective | ‖y − A f_θ(ε)‖² (LR measurement loss only) |
| Optimizers | two Adam optimizers (optax defaults), **θ lr 1e-5** on all 241 parameter leaves, **ε lr 1e-3**; one jitted joint step with donated buffers (`make_joint_step`, `scripts/run_sr4_iadrift_smoke.py`) |
| Steps | **1,000** per run; one run per (image, condition), deterministic, no restarts |
| Outputs | **two pre-declared checkpoints from the same run: step 100 and step 1,000**, each evaluated through the **eager** generator path |
| Numerics | `XLA_PYTHON_CLIENT_ALLOCATOR=platform` set before importing JAX (required for θ-gradients under WSL) |
| Monitoring (not used for any choice) | LR loss of the compiled step every 50 steps |
| Runs | 100 images × 2 conditions = **200 runs**, about 2.5 h on the RTX A4000 |

### 4.1 Decision record: learning rates and the step-100 checkpoint

**Decided 2026-10-05 by the project owner, before any IADrift run on test images.** Options offered: (a) plain IADrift with two checkpoints (100 and 1,000), (b) plain IADrift at 1,000 steps only, (c) wait a day for a constrained variant. Chosen: **(a)**. All evidence below is from **development images**.

| Setting | Decision | Development evidence |
|---|---|---|
| **θ lr = 1e-5** | the largest rate of the tuning grid | Tuning grid θ lr ∈ {1e-6, 3e-6, 1e-5} × ε lr ∈ {0, 1e-3}, 4 dev images, 1,000 steps (notebook 02, cells 0–13). The best mean PSNR gain over CSGM at any step was **+0.30 dB at step 50 for θ lr 1e-5**, vs +0.10 dB (3e-6, step 50) and +0.05 dB (1e-6, steps 100–150). Smaller rates don't avoid the later decline, they only slow it (one trade-off curve). θ lr 1e-5 is also the setting of all earlier IADrift runs (smoke test, notebook 02 first run). |
| **ε lr = 1e-3** | kept as in all earlier runs | ε hardly matters: ε lr 0 and 1e-3 gave nearly identical results at every θ lr (notebook 02); θ-only adaptation reproduced the joint results (notebook 05 / notebook 04 Check 2: 18.18 vs 18.29 dB at θ lr 1e-5). Kept nonzero so the method stays the joint (θ, ε) optimization IAGAN specifies. |
| **Checkpoint at step 100** (early-stopped variant) | pre-declared, not tuned | On development images PSNR peaked early and briefly and then fell at every setting: peak at **step 50** for θ lr 1e-5 (notebook 02); HR-best step 50 at both θ lrs in the 2,000-step θ-only runs (notebook 04, Check 2). The peak step is itself an HR-based (oracle) choice, so it can't be used as a stopping rule. **Step 100** was declared instead as a fixed, round early-stop point inside that early regime, with a margin beyond the noisy 50-step peak, and before any test result existed. |
| **Checkpoint at step 1,000** (as specified) | pre-declared | The length of the development IADrift runs; reports IADrift as specified without early stopping. |
| Both from **one run** | — | No extra compute; the two outputs are two declared views of the same optimization, so no step count is selected from test data. |

**What was deliberately not done:** no step count, learning rate or variant was chosen from the HR-best step on test images, and the constrained variants (weight anchor, frequency anchor, restricted layers) were not run before freezing.

## 5. Metrics (all against the 256×256 HR; final images from the eager path)

| Metric | Definition |
|---|---|
| **PSNR** (dB) | data range 2, unclipped; reported as mean of per-image PSNR and as PSNR of the mean MSE (IAGAN convention); clipped PSNR secondary |
| **LPIPS** | AlexNet backbone, `lpips` 0.1.4, inputs clipped to [-1, 1]; lower is better |
| Blockiness | `freq_metrics.blockiness`: mean pixel jump across 16-pixel tile borders ÷ mean jump at the other 4-pixel-aligned positions; ≈ 1 = no tile seams |
| Band SNR | `freq_metrics.band_snr_db`; bands low 0–0.0625, mid 0.0625–0.125, high 0.125–0.25, very high > 0.25 cycles/pixel |
| LR loss | ‖y − A x‖², eager |

## 6. Analysis plan

**Primary comparisons (co-primary metrics PSNR and LPIPS; correct label):**
1. IADrift @1000 + BP vs CSGM + BP
2. IADrift @100 + BP vs CSGM + BP

That's 2 comparisons × 2 metrics = **4 primary tests**. Paired over the 100 images: for each image i, the difference dᵢ = mᵢ(first method) − mᵢ(second method) of metric m; their mean d̄ with a normal-approximation 95% CI, d̄ ± 1.96 s/√n, where s is the sample standard deviation of the dᵢ (divisor n − 1) and n = 100 is the number of images, so s/√n is the standard error of d̄; Wilcoxon signed-rank p-value, count of images where the first method is better. **Holm correction across the 4 primary tests.**

**Secondary (reported, uncorrected, labelled exploratory):**
- the same comparisons without BP; and with the wrong label;
- IADrift (+ BP) vs bicubic + BP;
- label effect for IADrift (correct − wrong), raw and + BP;
- blockiness and band SNR for all methods; LR loss;
- visual examples: the first 4 images of the fixed sample (not chosen by result), full image + 80×80 detail crop.

## 7. Run procedure and failure handling

- Runner `scripts/run_sr4_iadrift_test.py` (written 2026-10-05; modelled on `run_sr4_csgm_test.py`): resumable 32-run chunks, manifest with a settings hash, `flock`, atomic writes, `--dry-run`, `--num-images N`. Overnight driver `scripts/run_overnight_iadrift_test.sh` (log `logs/sr4_iadrift_test_n100.log`). Run inside `tmux`, with notebook kernels closed and Windows sleep disabled.
- Outputs (full run): `phase2_sr4_iadrift_test_v1_n100_00ca59364e18.{jsonl,manifest.json}` and `…_arrays/img#####_{oracle,wrong}.npz` (outputs and ε at steps 100 and 1,000, loss trace). Each run also regenerates the CSGM start eagerly and records its maximum difference from the saved CSGM output (pipeline check).
- Before the full run: `--dry-run`, then a 1-image check (test image 1704, already disclosed) only to confirm the pipeline runs; its result is not used to change anything.
- Non-finite values or a crash: rerun the **identical** run (infrastructure retry). A method failure (e.g. divergence) is reported as a failure, not rerun with other settings.
- Results are analysed in a test notebook (notebook 03 style, or a new test notebook); notebook 04 only loads them in its read-only test section.

## 8. Deviations

- **2026-10-05, infrastructure only (method unchanged):** the first 32-run chunk of the full IADrift test run segfaulted inside the Python interpreter after 16 completed runs (exit 139). The 16 runs were saved atomically and kept. The driver was changed to **10-run chunks** and now counts a chunk as failed only if it saved no new run; the run resumed from run 17 with identical settings. Not motivated by any result.

## 9. Outcome (added after the run, 2026-10-05; does not change Sections 1–7)

The run completed as specified: 200/200 runs, all outputs finite, the regenerated CSGM starts identical to the saved ones. Analysis: `notebooks/03_csgm_test_analysis.ipynb`, Part 3 (cache `iadrift_test_postproc_v1/`, primary tests in `primary.json`).

**Primary results** (100 test images, correct label, Holm-corrected over the 4 tests; all significant):

| vs CSGM + BP | PSNR | LPIPS (lower is better) |
|---|---|---|
| IADrift @1000 + BP | −2.92 dB [−3.18, −2.66], Holm p 1.6e-17, better on 0/100 | +0.082 [+0.066, +0.097], Holm p 1.0e-14, better on 12/100 |
| IADrift @100 + BP | −0.12 dB [−0.20, −0.04], Holm p 1.4e-3, better on 33/100 | −0.008 [−0.013, −0.004], Holm p 3.8e-5, better on 72/100 |

**Reporting note.** Both checkpoints were declared in Section 4.1, so both are reported. The fact that step 100 turns out better than step 1,000 on the test set is a **result**, not a new choice. Any further work that treats the step-100 variant as "the" IADrift method (e.g. new experiments built on it) is a post-hoc decision informed by test results. It must be labelled as such and developed on development images only.

## 10. Development work after the test run (not part of this protocol)

After Section 9, further experiments were run **on development images only** (notebook 04, Checks 9–16): noise-label optimization and swaps, IADrift from better starts, one-image post-processing demos, posterior-mean ensembles of CSGM and IADrift runs, and high-frequency constraints (colour penalty, frequency anchor). None of them was run on the test set, and none changes Sections 1–9. They are exploratory (n = 1–6 images) and must be reported as such.

## 11. Addendum A (post-hoc, 2026-10-06): posterior-mean ensemble on the test set

**Status:** post-hoc. Motivated by development results (notebook 04 Check 13, 6 images) obtained after Section 9. Frozen before any run below; signed off by the project owner on **2026-10-06**. Does not change Sections 1–10.

- **Images:** the 100 test images of Section 1, **correct label only**.
- **Members (5 per image):** the saved CSGM test restarts r0, r1 (Section 3; `phase1_sr4_csgm_label_test_v2_n100_9e24c78a47ab_arrays/img#####_oracle_r{0,1}.npz`), plus 3 new frozen-CSGM runs (Section 3 settings: Adam lr 0.02, 5,000 steps, no regularizer) on y circularly shifted by (0,1), (1,0), (1,1) LR pixels, each started from the r0 initialization (`initialize(image, 0)`), output shifted back by 4 HR pixels per LR pixel. The optimization loop is imported unchanged from `scripts/run_sr4_csgm_test.py`.
- **Primary method Ens-W:** x̂ = BP(Σ wₖ xₖ), wₖ ∝ exp(−(Lₖ − L_min)/T), Lₖ = eager LR loss ‖y − A xₖ‖² of member k in original coordinates, T = max(std(L), 1e-6) (population std), weights normalized to sum 1. BP as in Section 3.
- **Secondary ensembles:** Ens-U (uniform mean of the 5), E1 (mean of r0, r1), E2 (mean of r0 + 3 shifts), each + BP.
- **Reference:** CSGM + BP as frozen in Section 3 (restart with the lowest final LR loss), recomputed and checked against `csgm_test_postproc_v1/`.
- **Primary tests (Holm over 2):** Ens-W + BP vs CSGM + BP, on PSNR and on LPIPS; paired over the 100 images, mean difference d̄ with a normal-approximation 95% CI, d̄ ± 1.96 s/√n (s = sample standard deviation of the 100 per-image differences, n = 100; as in Section 6), Wilcoxon signed-rank p, count of images better (as in Section 6).
- **Secondary (uncorrected, exploratory):** Ens-U, E1, E2 + BP vs CSGM + BP; every ensemble vs bicubic + BP; PSNR of mean MSE; blockiness; band SNR; LR losses; weight entropy / effective number of members; MMSE check (members' mean MSE ÷ Ens-U MSE before BP, in dB, vs the independent-sample prediction 10·log₁₀(2/1.2) = 2.22 dB, with caveats); the first 4 images of the fixed sample as visual examples.
- **Runner:** `scripts/run_sr4_ensemble_test.py`, settings hash **`ca2d0ec5e6ff`**, driver `scripts/run_overnight_ensemble_test.sh` (10-run chunks, log `logs/sr4_ensemble_test_n100.log`), output `csgm_test_ensemble_v1/`; analysis `notebooks/07_ensemble_test.ipynb` Part 1, cache `csgm_test_ensemble_postproc_v1/`.
- **Pre-launch checks (2026-10-06, passed):** `--repro-check` reproduced bit-exactly (final loss, ε, output, loss trace) the saved CSGM test run 1704/oracle/r0 (final loss 100.2454833984375) and the development shifted run 9780 shift (0,1) from notebook 04 Check 13 (107.30801391601562). Allocator: default (not `platform`).
- **Failure handling:** as in Section 7.

### 11.1 Outcome (added after the run, 2026-10-07; does not change Section 11)

The run completed as specified: 300/300 shifted runs, all finite; settings hash `ca2d0ec5e6ff` unchanged. Analysis: `notebooks/07_ensemble_test.ipynb` Part 1 (cache `csgm_test_ensemble_postproc_v1/`, primary tests in `primary.json`); presentation: `github/notebooks/03_ensemble_posterior_mean.ipynb`. The recomputed CSGM + BP reference is identical to `csgm_test_postproc_v1/`.

**Primary results** (100 test images, correct label, Holm over 2; both significant):

| vs CSGM + BP | PSNR | LPIPS (lower is better) |
|---|---|---|
| Ens-W + BP | +1.57 dB [+1.40, +1.74], Holm p 7.8e-18, better on 100/100 | −0.032 [−0.040, −0.024], Holm p 5.7e-10, better on 76/100 |

Secondary (exploratory): uniform Ens-U + BP +2.13 dB / LPIPS −0.037 vs CSGM + BP; MMSE check 2.33 dB measured vs 2.22 dB predicted.

**Deviation (infrastructure only, 2026-10-06):** the first launch was killed by the Linux OOM killer during run 3 (a concurrent smoke-test process used the memory); runs 1–2 had been saved atomically and were kept (deterministic); the driver was relaunched with identical settings. Not motivated by any result.
