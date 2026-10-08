# Archived side-experiment scripts (unported)

These four development experiments are mentioned in one line each in `notebooks/04_development_ablations.ipynb` §7
("also tried") and are **not** part of any reported table or figure, so they were not ported to the `iadrift/`
library. They are kept exactly as they were run, together with the original modules they import, for transparency:

| Script | Experiment |
|---|---|
| `run_z_opt_dev.py` | optimizing the noise labels z as a soft (relaxed) code, jointly with ε |
| `run_z_swap_dev.py` | single-coordinate swaps of z at the CSGM solution |
| `run_z_swap_early_dev.py` | z swaps early in CSGM |
| `run_combo_iadrift100_dev.py` | a better z start combined with IADrift @100 |
| `run_sr4_csgm_test.py`, `run_sr4_iadrift_smoke.py`, `run_sr4_iadrift_dev.py`, `freq_metrics.py` | the original modules they import (superseded by `iadrift/` and `scripts/test/`, `scripts/dev/`) |

They expect the original layout (`~/work/drifting`, `~/work/hf_cache`, `~/work/datasets/ImageNetV2`, hardcoded at the top
of `run_sr4_csgm_test.py`) and run from this directory: `cd scripts/dev_archive && python run_z_opt_dev.py`.
