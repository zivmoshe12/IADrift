# Development archive (raw working notebooks)

**These are the raw notebooks we worked in during the project, kept for transparency. Outputs are cleared** (the
largest was over 150 MB with its figures). They are not cleaned up: cell order follows the order the work was done in,
some checks were later superseded, and they import the project scripts from `~/work/IADrift/scripts` and read the
experiment caches from `~/work/datasets/ImageNetV2` (not part of this repository). For the results, read the
presentation notebooks in `notebooks/` instead. Every experiment in these notebooks that feeds a reported number has
been ported to a runner in `scripts/test/`, `scripts/dev/` or `scripts/analysis/` (each runner's docstring names the
cells it was ported from, and its `--dry-run` finds the cached outputs these notebooks wrote).

| Notebook | Content | Used by |
|---|---|---|
| `01_phase1_baseline.ipynb` | first CSGM baseline on a few images | — (superseded) |
| `02_iadrift_dev_tuning.ipynb` | IADrift learning-rate grid on 4 development images; basis of the step-100 checkpoint | notebook 1 §5 (choices), notebook 2 |
| `03_csgm_test_analysis.ipynb` | **the pre-registered test analysis** (CSGM, BP, LPIPS, IADrift; `docs/PROTOCOL.md` §9) | notebook 1 |
| `04_postproc_controls_dev.ipynb` | development Checks 1–16 (fusion, θ-only, generator diagnostics, ε projection, representation tests, z experiments, ensembles, penalties) | notebooks 2 and 4 |
| `05_theta_only_dev.ipynb` | θ-only adaptation runs | notebook 2 §3 |
| `06_noise_label_role_dev.ipynb` | what z and ε control (8 classes) | notebook 4 §1 |
| `07_ensemble_noisy_test.ipynb` | analysis of the post-hoc ensemble test run (`docs/PROTOCOL.md` §11) | notebook 3 |
