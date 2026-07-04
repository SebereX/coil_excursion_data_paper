"""Generate excursion_distance_scan.pdf using bundled REGCOIL and VMEC summary data."""

from pathlib import Path

from simsopt.mhd import Vmec
from regcoil_distance_scan import make_excursion_plot


def main():
    release_root = Path(__file__).resolve().parents[1]
    qh_root = release_root / "data" / "precise_QH"

    # Mirror the streamlined_regcoil_test.py plotting defaults.
    s_target = [1.0, 0.30, 0.12]
    thry_scale = 0.004
    y_lim_top = 0.15
    legend = False

    regcoil_run_folders = [
        str(qh_root / "regcoil_scan_results_precise_QH_s_target_1.00") + "/",
        str(qh_root / "regcoil_scan_results_precise_QH_s_target_0.30") + "/",
        str(qh_root / "regcoil_scan_results_precise_QH_s_target_0.12") + "/",
    ]

    vmec_files = [
        str(qh_root / "regcoil_scan_results_precise_QH_s_target_1.00" / "wout_downsampled_precise_QH_s_target_1.00.nc"),
        str(qh_root / "regcoil_scan_results_precise_QH_s_target_0.30" / "wout_downsampled_precise_QH_s_target_0.30.nc"),
        str(qh_root / "regcoil_scan_results_precise_QH_s_target_0.12" / "wout_downsampled_precise_QH_s_target_0.12.nc"),
    ]

    vmec = Vmec(vmec_files[0])
    a_minor_ref = vmec.wout.Aminor_p

    make_excursion_plot(
        regcoil_run_folders=regcoil_run_folders,
        vmec_coil_folder=str(qh_root) + "/",
        vmec_files=vmec_files,
        s_values=s_target,
        a_minor_ref=a_minor_ref,
        thry_scale=thry_scale,
        y_lim_top=y_lim_top,
        legend=legend,
        show=False,
    )

    print(f"Saved {qh_root / 'excursion_distance_scan.pdf'}")


if __name__ == "__main__":
    main()
