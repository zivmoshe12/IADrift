# IADrift: image-adaptive inversion of a drifting model for super-resolution

**Authors:** Aviram Ohayon, Ziv Moshe · Bar-Ilan University, course 83891 (Advanced topics in signal and image processing) ·
**Report:** REPORT_LINK

Does IAGAN-style adaptation (Hussein, Tirer & Giryes, AAAI 2020), which improves GAN priors for inverse problems by
adapting the generator's weights to each measurement, carry over when the prior is a **drifting model**
([Deng et al., 2026](https://arxiv.org/abs/2602.04770)) instead of a GAN? Both generate an image in a single forward
pass, but a drifting model is trained with a different objective and takes Gaussian noise of the same dimension as the
image (plus 32 discrete noise labels), not a low-dimensional latent code. We study one pretrained drifting model,
`pixel_B_sota`, as the prior for **super-resolution ×4** (64×64 → 256×256) of ImageNet-V2 images, against CSGM
(Bora et al., 2017) and bicubic upsampling. All conclusions are about this generator and this task.

This repository contains the code that produced every number in the report: a small library (`iadrift/`), one
command-line runner per experiment (`scripts/`), and notebooks that present the results from compact precomputed files
(`results/`).

## Main results (100 held-out test images)

| Method | PSNR (dB) ↑ | LPIPS ↓ |
|---|---:|---:|
| bicubic + BP | **25.57** | 0.358 |
| CSGM + BP | 22.75 | 0.284 |
| IADrift @100 steps + BP | 22.63 | 0.276 |
| IADrift @1000 steps + BP | 19.83 | 0.366 |
| *Post-hoc addendum:* weighted ensemble of 5 CSGM runs (E3w) + BP | 24.32 | **0.252** |

BP = back-projection onto the images consistent with the measurement. The first four rows are pre-registered
(`docs/PROTOCOL.md`, frozen before the IADrift test run); the ensemble was specified in a protocol addendum (§11)
before its test run. With this generator, CSGM is perceptually better than bicubic (LPIPS) but worse in PSNR;
adapting the weights keeps lowering the measurement loss while filling the unmeasured frequencies above the LR limit
with invented detail; averaging several reconstructions (an approximate posterior mean) removes much of it.

![Methods](figures/model_diagram_methods.png)

## Repository layout

```
iadrift/                 the library: operator and back-projection, generator calls, CSGM, IADrift, ensembles,
                         noisy SR, metrics, statistics, resumable checkpoints
scripts/data/            prepare_data.py: download ImageNet-V2, measurements, split, test sample
scripts/test/            the test-set runs (CSGM, IADrift, ensemble) and their overnight driver
scripts/dev/             the development experiments (development images only)
scripts/analysis/        evaluation of the raw runs (BP, LPIPS, statistics) and export of results/
notebooks/               presentation notebooks 01-04 (math -> code -> results), CPU only
notebooks/dev_archive/   the raw working notebooks of the project (outputs cleared), for transparency
results/                 compact precomputed results read by the notebooks
docs/PROTOCOL.md         the test protocol (frozen before the IADrift test run) with its addendum
figures/                 model and method diagrams
```

## Installation

Python 3.12, Linux.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                                   # notebooks only (CPU)
pip install -r requirements-experiments.txt                       # experiments (CUDA 12 GPU, ~16 GB)
git clone https://github.com/lambertae/drifting third_party/drifting
git -C third_party/drifting checkout accd0cf09c33b70892d33941d2a287ca86cb92e1   # the commit we used
pip install -r third_party/drifting/requirements.txt
```

Locations are set with environment variables (defaults in brackets):

| Variable | Meaning |
|---|---|
| `IADRIFT_DATA` [`data/ImageNetV2`] | ImageNet-V2, the derived measurement files, and every experiment cache |
| `DRIFTING_DIR` [`third_party/drifting`] | the upstream drifting code |
| `IADRIFT_HF_CACHE` [`hf_cache`] | the pretrained `pixel_B_sota` weights (downloaded from Hugging Face on first use) |

## Reproducing the results

Every runner has `--help`, a `--dry-run` (validate the plan, report what is already cached; writes nothing) and is
resumable. Runners that optimize have a `--smoke`/`--repro-check` that reruns part of a saved run and requires a
bit-exact match.

**1. Data** (~10 min):

```bash
python scripts/data/prepare_data.py            # ImageNet-V2 MatchedFrequency -> IADRIFT_DATA
python scripts/data/prepare_data.py --check    # verify (index, split, test sample, measurements)
```

**2. Test set** (pre-registered; RTX A4000 timings):

| Result (report / notebook) | Command | Runs, time |
|---|---|---|
| CSGM, correct and wrong label (Table 1) | `bash scripts/test/run_overnight.sh csgm` | 400, ~7.5 h |
| IADrift @100 / @1000 (Tables 1-2) | `bash scripts/test/run_overnight.sh iadrift` | 200, ~3 h |
| Posterior-mean ensemble (addendum, notebook 03) | `bash scripts/test/run_overnight.sh ensemble` | 300, ~5 h |
| BP, LPIPS, statistics | `python scripts/analysis/evaluate_test.py --part all` | CPU, ~30 min |
| `results/` for the notebooks | `python scripts/analysis/export_results.py`, `export_ensemble_results.py` | CPU |

**3. Development experiments** (development images only; one runner each, in `scripts/dev/`):

| Experiment | Runner | Notebook |
|---|---|---|
| CSGM settings sweep (100 images; chose lr 0.02, no regularizer) | `run_csgm_settings_sweep.py` | 01 §4 |
| IADrift learning rates and the step-100 checkpoint | `run_iadrift_dev.py` | 01 §5 |
| CSGM on the development images | `run_csgm_dev.py` | 02 |
| Fitting the HR image directly (CSGM-HR, IADrift-HR) | `run_representation_dev.py`, `run_hr_fit_dev.py` | 02 §1 |
| θ-only adaptation | `run_theta_only_dev.py` | 02 §3 |
| In-range SR, generator samples | `run_representation_dev.py`, `run_generator_samples_dev.py` | 02 §4-5 |
| What z and ε control; an image in place of ε | `run_noise_label_role_dev.py`, `run_image_in_eps_dev.py` | 04 §1-2 |
| ε kept on the Gaussian shell | `run_eps_projection_dev.py` | 04 §3 |
| Noise-label search | `run_noise_label_search_dev.py`, `run_z_swap_confirm_dev.py` | 04 §4 |
| Ensembles of CSGM and of IADrift runs | `run_ensemble_dev.py`, `run_iadrift_ensemble_dev.py` | 04 §5 |
| Colour penalty and frequency anchor | `run_hf_constraints_dev.py` | 04 §6 |
| Noisy SR ×4, σ = 10/255, Gaussian prior on ε (development sweep only) | `run_noisy_dev.py` | — |

Then `python scripts/analysis/export_dev_results.py` and `export_ablation_results.py` rebuild `results/`.

**4. Notebooks** (CPU, seconds): `jupyter lab notebooks/`. They read only `results/`.

## Numerics and verification

The generator computes in bf16 and the optimizations are chaotic, so a rerun reproduces a saved run bit-exactly only
with the same software and GPU settings. Each runner sets them itself (`iadrift.runtime.setup`):

- CSGM-type runs use JAX's default GPU allocator.
- Runs that differentiate w.r.t. the generator weights (IADrift, θ-only, HR fit, penalties) use
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (the default allocator hangs under WSL2) and
  `XLA_FLAGS=--xla_gpu_deterministic_ops=true`; without the flag, two identical runs can pick different GPU kernels
  and differ.
- Keep the two kinds of run in separate processes.

Every runner's `--smoke` (or `--repro-check`) reruns part of a saved run and compares it with the cache. On an RTX A4000
(JAX 0.4.37, CUDA 12) all of them match bit-exactly, except `scripts/dev/run_iadrift_dev.py`: that learning-rate grid
was the first IADrift run, made before the deterministic flag, so its cache is one draw of the kernel variation and
its smoke checks agreement within the observed run-to-run variation (≤ 0.1 dB PSNR). `prepare_data.py --check`,
`evaluate_test.py --check` and the `export_*.py --check` modes recompute the derived files and compare them with the
saved ones (identical).

## Data and license

[ImageNet-V2](https://github.com/modestyachts/ImageNetV2) (MatchedFrequency) is not included. `results/` contains, besides
metrics, a few 256×256 crops derived from it (the first 4 test images and a few development images, float16) so the
figures can be drawn; they are included for research and illustration only and remain subject to the ImageNet /
ImageNet-V2 terms of use. Code: MIT (`LICENSE`). The pretrained generator and its code belong to their authors.

## References

- A. Bora, A. Jalal, E. Price, A. G. Dimakis. *Compressed Sensing using Generative Models.* ICML 2017.
- S. A. Hussein, T. Tirer, R. Giryes. *Image-Adaptive GAN based Reconstruction.* AAAI 2020.
- M. Deng, H. Li, T. Li, Y. Du, K. He. *Generative Modeling via Drifting.* arXiv:2602.04770, 2026.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang. *The Unreasonable Effectiveness of Deep Features as a
  Perceptual Metric (LPIPS).* CVPR 2018.
- B. Recht, R. Roelofs, L. Schmidt, V. Shankar. *Do ImageNet Classifiers Generalize to ImageNet?* ICML 2019.
