# Scripts and data to reproduce main figures in the "Estimating coil features from an equilibrium" paper

This repository bundles scripts and datasets needed to reproduce the main figures in the paper "Estimating coil features from an equilibrium" by E. Rodriguez and W. Sengupta. The main figures are:
- `coil_set_k1.png` (Figure 1)
- `regcoil_phi_comparison_simple_true.png` (Figure 3)
- `excursion_distance_scan.pdf` (Figure 4)

All paths below are relative to the repository root.


## Included files

Root files:
- `README.md`
- `LICENSE`
- `requirements.txt`
- `.gitattributes`
- `.gitignore`

Script files:
- `scripts/streamlined_regcoil_test.py`
- `scripts/downsample_vmec.py`
- `scripts/regcoil_distance_scan.py`
- `scripts/regcoil_utils.py`
- `scripts/utils_surface_current.py`
- `scripts/make_coil_set_k1.py`
- `scripts/make_regcoil_phi_comparison_simple_true.py`
- `scripts/make_excursion_distance_scan.py`

Figure outputs:
- `figures/coil_set_k1.png`
- `figures/regcoil_phi_comparison_simple_true.png`
- `data/precise_QH/excursion_distance_scan.pdf`

Data folders:
- `data/precise_QH/`
- `data/precise_QA/`
- `data/W7X/`

### Included case families

- `precise_QH`
  - `data/precise_QH/regcoil_scan_results_precise_QH_s_target_0.12`
  - `data/precise_QH/regcoil_scan_results_precise_QH_s_target_0.30`
  - `data/precise_QH/regcoil_scan_results_precise_QH_s_target_1.00`
- `precise_QA`
  - `data/precise_QA/regcoil_scan_results_preciseQA`
  - `data/precise_QA/regcoil_scan_results_preciseQA_s0p12`
  - `data/precise_QA/regcoil_scan_results_preciseQA_s0p72`
- `W7X`
  - `data/W7X/regcoil_scan_results_d23p4_tm_s_target_0.12`
  - `data/W7X/regcoil_scan_results_d23p4_tm_s_target_0.30`
  - `data/W7X/regcoil_scan_results_d23p4_tm_s_target_1.00`

## Included scripts

- `scripts/streamlined_regcoil_test.py`
  - Main script to make the analysis of REGCOIL on surfaces at different distances.
  - By default, does not rerun VMEC/REGCOIL (uses bundled data).
- `scripts/downsample_vmec.py`
  - Helper for building downsampled VMEC inputs to consider configs at different radii.
- `scripts/regcoil_distance_scan.py`
- `scripts/regcoil_utils.py`
- `scripts/utils_surface_current.py`
- `scripts/make_coil_set_k1.py`
  - Recreates `figures/coil_set_k1.png`.
- `scripts/make_regcoil_phi_comparison_simple_true.py`
  - Recreates `figures/regcoil_phi_comparison_simple_true.png`.
- `scripts/make_excursion_distance_scan.py`
  - Recreates `data/precise_QH/excursion_distance_scan.pdf` from bundled `coil_excursion_results.pkl` files.

The utility scripts were slimmed for publication by removing non-essential top-level test/demo entry points while preserving the workflow and plotting paths used in this repository.

## Environment

Install Python dependencies:

```bash
pip install -r requirements.txt
```

This repository contains large binary data files. Install Git LFS before cloning/pushing:

```bash
git lfs install
```

External tools for full end-to-end reruns:
- VMEC (`xvmec2000`) for `--run-vmec`
- REGCOIL executable (`regcoil`) for `--run-regcoil`

If `regcoil` is not on your `PATH`, set:

```bash
export REGCOIL_EXECUTABLE=/absolute/path/to/regcoil
```

## Quick start (bundled-data mode)

From this folder:

```bash
python scripts/streamlined_regcoil_test.py
python scripts/make_coil_set_k1.py
python scripts/make_regcoil_phi_comparison_simple_true.py
python scripts/make_excursion_distance_scan.py
```

Expected outputs:
- `figures/coil_set_k1.png`
- `figures/regcoil_phi_comparison_simple_true.png`
- `data/precise_QH/excursion_distance_scan.pdf`

How `excursion_distance_scan.pdf` is made:
- `scripts/make_excursion_distance_scan.py` calls `make_excursion_plot(...)` in `scripts/regcoil_distance_scan.py`.
- It reads precomputed `coil_excursion_results.pkl` files from:
  - `data/precise_QH/regcoil_scan_results_precise_QH_s_target_1.00/`
  - `data/precise_QH/regcoil_scan_results_precise_QH_s_target_0.30/`
  - `data/precise_QH/regcoil_scan_results_precise_QH_s_target_0.12/`
- It reads VMEC reference excursions from:
  - `data/precise_QH/coil_excursion_results_vmec.pkl`

If you want to recompute those `coil_excursion_results.pkl` files (instead of using bundled ones), run `compute_coil_excursion_distance(load_or_compute="compute", folder=...)` for each run folder before generating the PDF.


## Optional full rerun

You may need additional VMEC `wout` inputs and external executables for a full rerun.

```bash
python scripts/streamlined_regcoil_test.py \
  --run-vmec --run-regcoil --run-regcoil-analysis \
  --vmec-executable /absolute/path/to/xvmec2000
```

Case-specific example for Precise QA data layout:

```bash
python scripts/streamlined_regcoil_test.py \
  --wout-name preciseQA \
  --target-dir ./data/precise_QA \
  --s-target 0.12 0.30 1.00 \
  --vmec-file ./data/precise_QA/regcoil_scan_results_preciseQA_s0p72/wout_downsampled_preciseQAs_target_0.72.nc
```

## Notes 

The bundle in this repository was done by compiling and downsizing scripts and data originally used. GitHub Copilot was used (with guidance) to achieve this.