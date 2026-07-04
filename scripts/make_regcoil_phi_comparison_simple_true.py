"""Generate regcoil_phi_comparison_simple_true.png using the existing comparison function."""

import argparse
from pathlib import Path

from regcoil_utils import comparison_anal_regcoil


def choose_default_inputs(release_root: Path):
    candidates = [
        ("regcoil_scan_results_preciseQA_s0p5", "wout_downsampled_preciseQAs_target_0.50.nc"),
        ("regcoil_scan_results_preciseQA_s0p72", "wout_downsampled_preciseQAs_target_0.72.nc"),
        ("regcoil_scan_results_preciseQA_s0p12", "wout_downsampled_preciseQAs_target_0.12.nc"),
    ]
    base = release_root / "data" / "precise_QA"
    for folder, wout_name in candidates:
        run_dir = base / folder
        vmec = run_dir / wout_name
        regcoil = run_dir / "regcoil_out.d0.0100_250x700.nc"
        if vmec.is_file() and regcoil.is_file():
            return vmec, regcoil
    raise FileNotFoundError(
        "No default Precise QA pair found. Expected one of: "
        "s0p5, s0p72, or s0p12 with both wout and regcoil_out files."
    )


def parse_args():
    release_root = Path(__file__).resolve().parents[1]
    default_vmec, default_regcoil = choose_default_inputs(release_root)
    default_output = release_root / "figures" / "regcoil_phi_comparison_simple_true.png"

    parser = argparse.ArgumentParser(description="Compare REGCOIL and analytic current potential.")
    parser.add_argument("--vmec-file", default=str(default_vmec), help="Path to VMEC wout file.")
    parser.add_argument("--regcoil-output", default=str(default_regcoil), help="Path to regcoil_out*.nc file.")
    parser.add_argument("--output", default=str(default_output), help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Show the figure window.")
    return parser.parse_args()


def main():
    args = parse_args()
    vmec_file = Path(args.vmec_file).resolve()
    regcoil_output = Path(args.regcoil_output).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not vmec_file.is_file():
        raise FileNotFoundError(f"VMEC file not found: {vmec_file}")
    if not regcoil_output.is_file():
        raise FileNotFoundError(f"REGCOIL output file not found: {regcoil_output}")

    comparison_anal_regcoil(
        vmec_file=str(vmec_file),
        regcoil_output_file=str(regcoil_output),
        output_file=str(output),
        show=args.show,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
