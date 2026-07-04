
"""
Generate VMEC input files and run many VMEC jobs from Python.

This script supports two levels of parallelism:
1) MPI ranks inside one VMEC case (ranks_per_case)
2) Several VMEC cases at once (max_concurrent_cases)
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import subprocess

from downsample_vmec import create_downsampled_cases


def parse_args():
    """Parse command-line options for VMEC batch runs."""
    default_target = Path(__file__).resolve().parents[1] / "data" / "precise_QH"
    default_vmec = default_target / "regcoil_scan_results_precise_QH_s_target_1.00" / "wout_downsampled_precise_QH_s_target_1.00.nc"

    parser = argparse.ArgumentParser(description="Generate and run multiple VMEC equilibria.")
    parser.add_argument("--ranks", type=int, default=4, help="MPI ranks per VMEC case.")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=2,
        help="Maximum number of VMEC cases to run concurrently.",
    )
    parser.add_argument(
        "--mpi-launcher",
        default="/usr/bin/mpirun",
        help="Path to mpirun/mpiexec launcher.",
    )
    parser.add_argument(
        "--vmec-executable",
        default="xvmec2000",
        help="Path to xvmec2000 executable.",
    )
    parser.add_argument(
        "--wout-name",
        default="precise_QH",
        help="Name tag used in generated folder names (e.g. precise_QH).",
    )
    parser.add_argument(
        "--vmec-file",
        default=str(default_vmec),
        help="Reference VMEC wout file used to compute a_minor and optionally create downsampled inputs.",
    )
    parser.add_argument(
        "--target-dir",
        default=str(default_target),
        help="Directory containing/receiving regcoil_scan_results_* folders.",
    )
    parser.add_argument(
        "--s-target",
        nargs="+",
        type=float,
        default=[0.12, 0.30, 1.00],
        help="Target s values for each case.",
    )
    parser.add_argument("--run-vmec", action="store_true", help="Run VMEC for generated cases.")
    parser.add_argument("--run-vmec-analysis", action="store_true", help="Run VMEC excursion analysis.")
    parser.add_argument("--run-regcoil", action="store_true", help="Run REGCOIL scan for each case.")
    parser.add_argument("--run-regcoil-analysis", action="store_true", help="Compute coil excursion from REGCOIL outputs.")
    return parser.parse_args()


def build_vmec_command(vmec_executable, input_name, ranks_per_case, mpi_launcher, extra_mpi_args):
    """Create the command for one VMEC run."""
    if ranks_per_case > 1:
        return [mpi_launcher, "-np", str(ranks_per_case), *extra_mpi_args, vmec_executable, input_name]
    return [vmec_executable, input_name]


def run_single_vmec(
    input_file,
    vmec_executable,
    ranks_per_case=1,
    mpi_launcher="mpirun",
    extra_mpi_args=None,
    mpi_env=None,
):
    """Run one VMEC case and write stdout/stderr to a case log file."""
    input_path = Path(input_file).resolve()

    # Allow passing either a VMEC input file or a case directory.
    if input_path.is_dir():
        candidates = sorted(input_path.glob("input*"))
        if not candidates:
            raise FileNotFoundError(f"No VMEC input file matching 'input*' in {input_path}")
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple VMEC input files found in {input_path}: "
                + ", ".join(p.name for p in candidates)
                + ". Please keep only one or pass an explicit input file path."
            )
        input_path = candidates[0]

    run_dir = input_path.parent
    log_file = run_dir / f"{input_path.name}.log"

    extra_mpi_args = extra_mpi_args or []
    cmd = build_vmec_command(
        vmec_executable=str(vmec_executable),
        input_name=input_path.name,
        ranks_per_case=ranks_per_case,
        mpi_launcher=mpi_launcher,
        extra_mpi_args=extra_mpi_args,
    )

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    if mpi_env:
        env.update(mpi_env)

    with open(log_file, "w") as fh:
        result = subprocess.run(
            cmd,
            cwd=run_dir,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )

    return {
        "input": str(input_path),
        "log": str(log_file),
        "returncode": result.returncode,
        "command": " ".join(cmd),
    }


def run_vmec_batch(
    input_files,
    vmec_executable,
    ranks_per_case=1,
    max_concurrent_cases=1,
    mpi_launcher="mpirun",
    extra_mpi_args=None,
    mpi_env=None,
):
    """Run many VMEC cases with bounded concurrency."""
    results = []
    with ThreadPoolExecutor(max_workers=max_concurrent_cases) as pool:
        futures = {
            pool.submit(
                run_single_vmec,
                file,
                vmec_executable,
                ranks_per_case,
                mpi_launcher,
                extra_mpi_args,
                mpi_env,
            ): file
            for file in input_files
        }

        for future in as_completed(futures):
            outcome = future.result()
            results.append(outcome)
            status = "OK" if outcome["returncode"] == 0 else "FAIL"
            print(f"[{status}] {outcome['input']} -> {outcome['log']}")

    return results


def main():
    args = parse_args()
    no_vmec_run = not args.run_vmec
    no_vmec_analysis = not args.run_vmec_analysis
    no_regcoil_run = not args.run_regcoil
    no_regcoil_aanalysis = not args.run_regcoil_analysis

    # Inputs for case generation
    wout_name = args.wout_name
    s_target = args.s_target
    target_dir = str(Path(args.target_dir).resolve()) + "/"
    thry_scale = 0.004; y_lim_top = 0.15; N_scan = 20; max_dist = 0.17; legend = False
    # thry_scale = 0.03; y_lim_top = 0.35; N_scan = 20; max_dist = 0.35; legend = False

    # VMEC launcher settings
    vmec_executable = args.vmec_executable
    ranks_per_case = args.ranks
    max_concurrent_cases = args.max_cases
    mpi_launcher = args.mpi_launcher
    extra_mpi_args = []

    # Default to the OpenMPI stack linked by xvmec2000 on this machine.
    mpi_env = {
        "PATH": "/usr/bin:" + os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", ""),
    }

    # VMEC file
    vmec_file = str(Path(args.vmec_file).resolve())
    if not os.path.isfile(vmec_file):
        raise FileNotFoundError(f"VMEC file not found: {vmec_file}")
    
    from simsopt.mhd import Vmec, Boozer 
    vmec = Vmec(vmec_file)
    A_minor = vmec.wout.Aminor_p

    # Z excursion of VMEC surfaces
    from utils_surface_current import scan_excursion_s
    if not no_vmec_analysis:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        scan_excursion_s(vmec_file = vmec_file, folder = target_dir, plot = False)

    # Create new VMEC input files
    folder_list = create_downsampled_cases(
        wout_name,
        s_target,
        target_dir=target_dir,
        only_folders=no_vmec_run,
        vmec_file=vmec_file,
    )

    if not no_vmec_run:
        # Run VMEC for all generated inputs
        results = run_vmec_batch(
            folder_list,
            vmec_executable=vmec_executable,
            ranks_per_case=ranks_per_case,
            max_concurrent_cases=max_concurrent_cases,
            mpi_launcher=mpi_launcher,
            extra_mpi_args=extra_mpi_args,
            mpi_env=mpi_env,
        )

        failures = [r for r in results if r["returncode"] != 0]
        if failures:
            print("\nSome VMEC runs failed:")
            for f in failures:
                print(f"  - input: {f['input']}")
                print(f"    log:   {f['log']}")
                print(f"    cmd:   {f['command']}")
            raise RuntimeError(f"{len(failures)} VMEC case(s) failed.")

        print("\nAll VMEC runs completed successfully.")

    # Run REGCOIIL for each case
    import numpy as np  
    distances = np.insert(np.linspace(0.015, max_dist, N_scan), 0, 0.01)
    vmec_file_list = []
    for folder in folder_list:
        # Find folder
        run_dir = Path(folder)
        
        # Find the wout file in the run directory
        wout_files = list(run_dir.glob("wout_*.nc"))
        if not wout_files:
            print(f"No wout file found in {run_dir}, skipping REGCOIL.")
            continue
        wout_file = wout_files[0]  # Assuming one wout file per case
        vmec_file_list.append(str(wout_file))
        
        if not no_regcoil_run:
            from regcoil_distance_scan import run_scan_distance
            run_scan_distance(wout_file, run_dir, load = False, distances = distances, d_crit = 0.05)

    # Post-process REGCOIL results here if needed
    from regcoil_distance_scan import compute_coil_excursion_distance
    if not no_regcoil_aanalysis:
        for folder in folder_list:
            compute_coil_excursion_distance(load_or_compute = "compute", folder = folder, distances = distances, plot = False)

    # Make plot
    import matplotlib.pyplot as plt
    from regcoil_distance_scan import make_excursion_plot
    make_excursion_plot(regcoil_run_folders = folder_list, vmec_coil_folder = target_dir, vmec_files = vmec_file_list, s_values = s_target, a_minor_ref = A_minor, thry_scale = thry_scale, y_lim_top = y_lim_top, legend = legend)
    plt.show()


if __name__ == "__main__":
    main()


