# Scripts and data to reproduce main figures in the "Estimating coil features from an equilibrium" paper

This folder packages the key scripts and data needed to reproduce the figures in the paper titled ¨Estimating coil features from an equilibrium¨ by E. Rodriguez and W. Sengupta. The main figures are
- `coil_set_k1.png`
- `regcoil_phi_comparison_simple_true.png`
and the scripts to generate them are included

Validated default figure inputs in this bundle use the first available Precise QA case in this order: `s0p5`, `s0p72`, `s0p12`.

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

Figure outputs:
- `figures/coil_set_k1.png`
- `figures/regcoil_phi_comparison_simple_true.png`

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
  - Portable version of the batch workflow.
  - Uses relative defaults inside this bundle.
  - By default, does not rerun VMEC/REGCOIL (uses bundled data).
- `scripts/downsample_vmec.py`
  - Helper for building downsampled VMEC inputs.
- `scripts/regcoil_distance_scan.py`
- `scripts/regcoil_utils.py`
- `scripts/utils_surface_current.py`
- `scripts/make_coil_set_k1.py`
  - Recreates `figures/coil_set_k1.png`.
- `scripts/make_regcoil_phi_comparison_simple_true.py`
  - Recreates `figures/regcoil_phi_comparison_simple_true.png`.

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
```

Expected outputs:
- `figures/coil_set_k1.png`
- `figures/regcoil_phi_comparison_simple_true.png`

The two figure scripts default to the first available case among `s0p5`, `s0p72`, and `s0p12`.
In the current repository state this resolves to:
- `data/precise_QA/regcoil_scan_results_preciseQA_s0p72/wout_downsampled_preciseQAs_target_0.72.nc`
- `data/precise_QA/regcoil_scan_results_preciseQA_s0p72/regcoil_out.d0.0100_250x700.nc`

## Optional full rerun

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

## Publishing

### GitHub push checklist

```bash
git init
git lfs install
git add .
git commit -m "Initial reproducibility bundle"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

If push is rejected with `non-fast-forward`:

```bash
git fetch origin
git pull --rebase origin main
git push origin main
```

### DOI

Use Zenodo to mint a DOI from a GitHub release tag (for example `v1.0.0`).
