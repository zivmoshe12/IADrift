# Earlier scripts

The full report lists four ideas under "Also tried" that we ran briefly on a few development images and do not report
numbers for. These are the scripts they ran with, unchanged apart from their docstrings:

| Idea in the report | Script |
|---|---|
| Frequency fusion | `fuse_with_bicubic` in `freq_metrics.py` |
| Soft style labels | `run_z_opt_dev.py` |
| Label swaps | `run_z_swap_dev.py`, `run_z_swap_early_dev.py` |
| IADrift@100 from a better label draw | `run_combo_iadrift100_dev.py` |

`run_sr4_csgm_test.py`, `run_sr4_iadrift_smoke.py`, `run_sr4_iadrift_dev.py` and `freq_metrics.py` are the older
modules these scripts import; `iadrift/` and the runners in `scripts/test/` and `scripts/dev/` replaced them.

The scripts are not part of the reproduction path. They use the folder layout we had at the time (`~/work/drifting`,
`~/work/hf_cache`, `~/work/datasets/ImageNetV2`, set at the top of `run_sr4_csgm_test.py`) and run from this folder,
e.g. `cd scripts/dev_archive && python run_z_opt_dev.py`.
