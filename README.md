# Streamlined REGCOIL Reproducibility Bundle

This folder packages the key scripts and data needed to reproduce the workflow around `streamlined_regcoil_test.py` and generate the two paper figures:
- `coil_set_k1.png`
- `regcoil_phi_comparison_simple_true.png`

## Folder layout

- `scripts/`: Python scripts used in the workflow.
- `data/precise_QH/`: bundled VMEC/REGCOIL data and result folders for the precise QH workflow.
- `data/precise_QA/`: bundled Precise QA REGCOIL scan folders and reference excursion files.
- `data/W7X/`: bundled W7X REGCOIL scan folders and reference excursion files.
- `data/W7X_short/`: bundled shortened W7X scan set used for faster comparisons.
- `figures/`: figure outputs (and default target location for regenerated figures).

### Included case families

- `precise_QH`
  - `data/precise_QH/regcoil_scan_results_precise_QH_s_target_0.12`
  - `data/precise_QH/regcoil_scan_results_precise_QH_s_target_0.30`
  - `data/precise_QH/regcoil_scan_results_precise_QH_s_target_1.00`
- `precise_QA`
  - `data/precise_QA/regcoil_scan_results_preciseQA`
  - `data/precise_QA/regcoil_scan_results_preciseQA_s0p12`
  - `data/precise_QA/regcoil_scan_results_preciseQA_s0p3`
  - `data/precise_QA/regcoil_scan_results_preciseQA_s0p5`
  - `data/precise_QA/regcoil_scan_results_preciseQA_s0p72`
- `W7X`
  - `data/W7X/regcoil_scan_results_d23p4_tm_s_target_0.12`
  - `data/W7X/regcoil_scan_results_d23p4_tm_s_target_0.30`
  - `data/W7X/regcoil_scan_results_d23p4_tm_s_target_1.00`
- `W7X_short`
  - `data/W7X_short/regcoil_scan_results_d23p4_tm_s_target_0.12`
  - `data/W7X_short/regcoil_scan_results_d23p4_tm_s_target_0.30`
  - `data/W7X_short/regcoil_scan_results_d23p4_tm_s_target_1.00`

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

## Environment

Install Python dependencies:

```bash
pip install -r requirements.txt
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
  --vmec-file ./data/precise_QA/regcoil_scan_results_preciseQA_s0p3/wout_downsampled_preciseQAs_target_0.30.nc
```

## Notes 

The bundle in this repository was done by compiling and downsizing scripts and data originally used. GitHub Copilot was used (with guidance) to achieve this.
