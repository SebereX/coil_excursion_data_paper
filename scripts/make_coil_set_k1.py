"""Generate coil_set_k1.png using the existing 3D plotting function."""

import argparse
from pathlib import Path

from utils_surface_current import make_nice_3d_plot_coils


def choose_default_vmec(release_root: Path):
    candidates = [
        ("regcoil_scan_results_preciseQA_s0p5", "wout_downsampled_preciseQAs_target_0.50.nc"),
        ("regcoil_scan_results_preciseQA_s0p72", "wout_downsampled_preciseQAs_target_0.72.nc"),
        ("regcoil_scan_results_preciseQA_s0p12", "wout_downsampled_preciseQAs_target_0.12.nc"),
    ]
    base = release_root / "data" / "precise_QA"
    for folder, wout_name in candidates:
        vmec = base / folder / wout_name
        if vmec.is_file():
            return vmec
    raise FileNotFoundError(
        "No default Precise QA wout file found. Expected one of s0p5, s0p72, or s0p12."
    )


def parse_args():
    release_root = Path(__file__).resolve().parents[1]
    default_vmec = choose_default_vmec(release_root)
    default_output = release_root / "figures" / "coil_set_k1.png"

    parser = argparse.ArgumentParser(description="Render coil set and principal curvature map.")
    parser.add_argument("--vmec-file", default=str(default_vmec), help="Path to VMEC wout file.")
    parser.add_argument("--output", default=str(default_output), help="Output PNG path.")
    parser.add_argument("--s", type=float, default=0.9, help="Surface label for k1 visualization.")
    parser.add_argument("--n-coils", type=int, default=40, help="Number of coils to draw.")
    return parser.parse_args()


def main():
    args = parse_args()
    vmec_file = Path(args.vmec_file).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not vmec_file.is_file():
        raise FileNotFoundError(f"VMEC file not found: {vmec_file}")

    make_nice_3d_plot_coils(
        vmec_file=str(vmec_file),
        output_file=str(output),
        s_plot=args.s,
        n_coils=args.n_coils,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
