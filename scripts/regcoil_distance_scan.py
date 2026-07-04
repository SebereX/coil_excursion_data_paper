import os
import numpy as np
import pickle
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from regcoil_utils import read_regcoil_output, solve_merkel_problem, make_current_density_structure
from utils_surface_current import evaluate_contour_RZ, load_vmec, max_excursion, vmec_compute_geometry, evaluate_on_grid, Struct, evaluate_surface_geometry, characterise_curve, make_coils_from_vmec


def run_scan_distance(vmec_file, output_dir, load = False, distances = None, d_crit = 0.05, regcoil_omp_threads=16):
    #############
    # LOAD VMEC #
    #############
    # Vmec file
    # vmec_file = "/home/erodrigu/Python/play_coils/regcoil_scan_results_preciseQA_s0p12/wout_downsampled_preciseQAs_target_0.12.nc"

    # Load VMEC geometry
    vs = load_vmec(vmec_file)
    s = [1.0]
    results = vmec_compute_geometry(vs, s, [0], [0])
    mu0 = 4*np.pi*1e-7
    G_current = results.G_Boozer[0] * 2*np.pi / mu0

    ########################
    # COMPUTE REGCOIL SCAN #
    ########################
    # Define parameters for the scan
    if distances is None or len(distances) == 0:
        N_scan = 20
        # distances = np.linspace(0.01, 0.25, N_scan) # 1.0
        # distances = np.linspace(0.01, 0.2, N_scan) # 0.5
        # distances = np.linspace(0.01, 0.2, N_scan) # 0.72
        # distances = np.linspace(0.01, 0.17, N_scan) # 0.3
        distances = np.insert(np.linspace(0.015, 0.17, N_scan), 0, 0.01) # 0.12
        N_scan = len(distances)
    else:
        N_scan = len(distances)


    # Directories for loading/saving results
    output_dir = Path(output_dir)
    ntheta_coil_r = 250
    nzeta_coil_r = 700
    ntheta_coil_f = 150
    nzeta_coil_f = 400
    # output_dir = "./regcoil_scan_results_preciseQA_s0p12/"

    # Run scans
    results = []
    with tqdm(total=N_scan, desc="Running distance scan") as pbar:
        for d in distances:
            if d < d_crit:
                ntheta_coil = ntheta_coil_r
                nzeta_coil = nzeta_coil_r
            else:
                ntheta_coil = ntheta_coil_f
                nzeta_coil = nzeta_coil_f
            base_filename = f"d{d:.4f}_{ntheta_coil}x{nzeta_coil}"
            try:
                result = solve_merkel_problem(vmec_file, 
                                            plasma_coil_distance=d, 
                                            net_poloidal_current=G_current, net_toroidal_current=0.0, 
                                            ntor_potential=20, mpol_potential=20, ntheta_coil=ntheta_coil, nzeta_coil=nzeta_coil, 
                                            save = True, base_filename=base_filename, output_dir = output_dir, load = load,
                                            regcoil_omp_threads=regcoil_omp_threads)
                results.append(result)
                print(f"Completed REGCOIL run for distance {d:.2f}. chi2_B = {result['chi2_B']:.3e}, max_Bnormal = {result['max_Bnormal']:.3e}")
            except:
                log_file = Path("./temp/regcoil_run_temp.log")
                print(f"REGCOIL run failed for distance {d:.2f}. Check the log file '{log_file}' for details.")
                results.append(None)  # Append None or some placeholder to indicate failure
            pbar.update(1)
    
    ########################
    # SAVE RESULTS TO DISK #
    ########################
    # # Load .pkl if it exists, to avoid overwriting previous results
    # if os.path.exists(output_dir + "scan_results.pkl"):
    #     with open(output_dir + "scan_results.pkl", "rb") as f:
    #         existing_results = pickle.load(f)
    #     # Append new results to existing results
    #     if isinstance(existing_results, list):
    #         results = existing_results + results
    #     else:
    #         print("Warning: existing scan results are not in expected format (list). Overwriting with new results.")


    # Save list of dictionaries 
    with open(output_dir / "scan_results.pkl", "wb") as f:
        pickle.dump(results, f)

def read_all_outputs(folder = "./regcoil_scan_results_preciseQA_s0p72/"):
    ############################
    # GET LIST OF OUTPUT FILES #
    ############################
    # Directory
    output_dir = folder

    # List of output files
    output_files = [f for f in os.listdir(output_dir) if f.endswith(".nc")]
    # Pop any wout file
    output_files = [f for f in output_files if not (f.startswith("wout") or f.startswith("jxbout"))]
    print(f"Found {len(output_files)} output files in '{output_dir}'.")

    # Extract distances from filenames
    distances = []
    for f in output_files:
        try:
            d_str = f.split("d")[1].split("_")[0]
            distances.append(float(d_str))
        except:
            print(f"Could not extract distance from filename '{f}'. Skipping this file.")
            distances.append(None)  # Append None or some placeholder to indicate failure

    # Sort files by distance
    sorted_files = [f for _, f in sorted(zip(distances, output_files))]

    ####################################
    # READ AND EXTRACT DATA FROM FILES #
    ####################################
    all_data = []
    for f in sorted_files:
        output_file_name = os.path.join(output_dir, f)
        data = read_regcoil_output(output_file_name, verbose = False)

        # Select lambda = 0.0
        if 'lambda' in data and len(data['lambda']) > 1:
            lambda_index = np.argmin(np.abs(data['lambda'] - 0.0))
        else:
            lambda_index = 0

        # Select only relevant data to return
        relevant_keys_nl = ['mnmax_coil', 'xm_coil', 'xn_coil', 
                            'rmnc_coil', 'zmns_coil',
                        'mnmax_potential', 'xm_potential', 'xn_potential', 
                            'net_poloidal_current_Amperes', 'net_toroidal_current_Amperes',
                            ]
        relevant_keys_l = ['single_valued_current_potential_mn',
                        'chi2_B', 'max_Bnormal'
                        ]

        data_filt = {key: data[key] for key in relevant_keys_nl if key in data}
        data_filt.update({key: np.squeeze(data[key][lambda_index, ...]) for key in relevant_keys_l if key in data})

        # Add to all_data
        all_data.append(data_filt)

    return all_data

def compute_coil_excursion_distance(load_or_compute = "compute", folder = "./regcoil_scan_results_preciseQA_s0p72/", distances = None, plot = False):
    if load_or_compute == "compute":
        # Load results
        results = read_all_outputs(folder)

        # Define parameters for the scan
        if distances is None:
            N_scan = 20
            # distances = np.linspace(0.01, 0.25, N_scan) # 1.0
            # distances = np.linspace(0.01, 0.2, N_scan) # 0.5
            # distances = np.linspace(0.01, 0.2, N_scan) # 0.72
            # distances = np.linspace(0.01, 0.17, N_scan) # 0.3
            distances = np.insert(np.linspace(0.015, 0.17, N_scan), 0, 0.01) # 0.12
        N_scan = len(distances)


        # Compute coil excursion distance for each case
        max_excursions = []
        max_curvatures = []
        max_geo_curvatures = []
        chi2_B_values = []
        max_Bnormal_values = []

        import signal

        class TimeoutException(Exception):
            pass

        def handler(signum, frame):
            raise TimeoutException()

        signal.signal(signal.SIGALRM, handler)
        TIMEOUT_SECONDS = 4  # Set your desired timeout per iteration (seconds)

        with tqdm(total=N_scan, desc="Computing coil excursions") as pbar:
            for idx, res in enumerate(results):
                try:
                    signal.alarm(TIMEOUT_SECONDS)
                    if res is not None:
                        potential_data = make_current_density_structure(res)
                        coils, _ = make_coils_from_vmec(None, None, N_coils=50, N_theta=200, full_info=True, no_B=True, parallel=True, Phi_data=potential_data)
                        exc_coils = []
                        curv_coils = []
                        geod_curv_coils = []
                        for j in coils:
                            X_c = coils[j]["X"]
                            Y_c = coils[j]["Y"]
                            Z_c = coils[j]["Z"]
                            curvature_c = coils[j]["curvature"]
                            geodesic_curvature_c = coils[j]["geodesic_curvature"]
                            max_curv = np.max(curvature_c)
                            max_geo_curv = np.max(np.abs(geodesic_curvature_c))
                            exc_coil, _, _ = max_excursion(X_c, Y_c, Z_c, verbose=False)
                            exc_coils.append(exc_coil)
                            curv_coils.append(max_curv)
                            geod_curv_coils.append(max_geo_curv)
                        max_excursions.append(np.max(exc_coils))
                        max_curvatures.append(np.max(curv_coils))
                        max_geo_curvatures.append(np.nanmax(geod_curv_coils))
                        chi2_B = res['chi2_B'] if 'chi2_B' in res else None
                        max_Bnormal = res['max_Bnormal'] if 'max_Bnormal' in res else None
                        chi2_B_values.append(chi2_B)
                        max_Bnormal_values.append(max_Bnormal)
                        print(f"d={distances[len(max_excursions)-1]:.2f}, max excursion: {max_excursions[-1]:.3e}," \
                              f" max curvature: {max_curvatures[-1]:.3e}, max geodesic curvature: {max_geo_curvatures[-1]:.3e}, " \
                              f"chi2_B: {chi2_B:.3e}, max_Bnormal: {max_Bnormal:.3e}")
                    else:
                        max_excursions.append(np.nan)
                        max_curvatures.append(np.nan)
                        max_geo_curvatures.append(np.nan)
                        chi2_B_values.append(np.nan)
                        max_Bnormal_values.append(np.nan)

                except (TimeoutException, Exception) as e:
                    print(f"Timeout for d={distances[idx]:.2f}. Skipping this case.")
                    max_excursions.append(np.nan)
                    max_curvatures.append(np.nan)
                    max_geo_curvatures.append(np.nan)
                    chi2_B_values.append(np.nan)
                    max_Bnormal_values.append(np.nan)

                finally:
                    signal.alarm(0)
                pbar.update(1)


        # Save results to disk
        with open(folder + "coil_excursion_results.pkl", "wb") as f:
            pickle.dump({
                'distances': distances,
                'max_excursions': max_excursions,
                'max_curvatures': max_curvatures,
                'max_geo_curvatures': max_geo_curvatures,
                'chi2_B_values': chi2_B_values,
                'max_Bnormal_values': max_Bnormal_values
            }, f)
        
        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(distances, max_excursions, 'ko-', label='Max Excursion')
            plt.ylabel('Max Excursion')
            ax2 = plt.gca().twinx()
            ax2.plot(distances, np.array(max_curvatures), 'ko:', label='Max Curvature')
            ax2.plot(distances, np.array(max_geo_curvatures), 'ks--', label='Max Geodesic Curvature')
            ax2.set_ylabel('Max Curvature')
            plt.xlabel('s')
            plt.tight_layout()
            plt.show()

    elif load_or_compute == "load":
        with open(folder + "coil_excursion_results.pkl", "rb") as f:
            data = pickle.load(f)
        distances = data['distances']
        max_excursions = data['max_excursions']
        max_curvatures = data['max_curvatures']
        max_geo_curvatures = data['max_geo_curvatures']

        with open("./coil_excursion_results_vmec.pkl", "rb") as f:
            data_vmec = pickle.load(f)
            # "s_values": s_values,
            # "max_excursions": max_excursions,
            # "max_curvatures": max_curvatures,
            # "max_geo_curvatures": max_geo_curvatures
        s_values = data_vmec["s_values"]
        max_excursions_vmec = data_vmec["max_excursions"]
        max_curvatures_vmec = data_vmec["max_curvatures"]
        max_geo_curvatures_vmec = data_vmec["max_geo_curvatures"]

        max_excursions_small = [1.346e-02, 1.974e-02, 2.690e-02, 3.516e-02, 4.487e-02, 1.065e-01]
        max_geo_curvatures_small = [3.151e+00, 3.372e+00, 3.677e+00, 9.234e+00, 1.320e+02, 2.212e+03]
        max_curvatures_small = [3.780e+01, 2.249e+01, 1.603e+01, 1.260e+01, 1.320e+02, 2.212e+03]
        N_eff = len(max_excursions_small)
        distances_small = (1 + distances[:N_eff]) * np.sqrt(0.5)

        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(1+distances[:-1], max_excursions[:-1], 'ko-', label='Max Excursion')
            # Fit a quadratic curve to the max excursion data (ignoring the last point which is an outlier)
            coeffs = np.polyfit(np.sqrt(s_values), max_excursions_vmec, 2)
            full_domain = np.linspace(0.5, 1.25, 100)
            fit_curve = np.polyval(coeffs, full_domain)
            plt.plot(full_domain, fit_curve, 'k:', label='Quadratic Fit')
            plt.plot(np.sqrt(s_values), max_excursions_vmec, 'ro-', label='Max Excursion VMEC')
            plt.plot(distances_small, max_excursions_small, 'bs-', label='Max Excursion Small')
            # plt.plot(distances, 0.04*10**(4*distances), 'r--', label='4*exp(d)')
            plt.ylabel('Max Excursion')
            # ax2 = plt.gca().twinx()
            # ax2.plot(distances, np.array(max_curvatures), 'ko:', label='Max Curvature')
            # ax2.plot(distances, np.array(max_geo_curvatures), 'ks--', label='Max Geodesic Curvature')
            # ax2.set_ylabel('Max Curvature')
            plt.xlabel('$$d$$')
            # plt.yscale('log')
            plt.tight_layout()
            plt.show()

def make_excursion_plot(regcoil_run_folders = None, vmec_coil_folder = None, vmec_files = None, s_values = None, a_minor_ref = 0.1679021975036401, thry_scale = 0.005, y_lim_top = 0.15, show = False, legend = True):
    """
    Make plot including all the excursion REGCOIL runs.
    """
    # REGCOIL results
    if regcoil_run_folders is None:
        regcoil_run_folders = ["./regcoil_scan_results_preciseQA/",
                            #    "./regcoil_scan_results_preciseQA_s0p5/", 
                            #    "./regcoil_scan_results_preciseQA_s0p72/",
                                "./regcoil_scan_results_preciseQA_s0p3/",
                                "./regcoil_scan_results_preciseQA_s0p12/"
                            ]
    
    if vmec_coil_folder is None:
        vmec_coil_folder = "./"
    if vmec_files is None:
        vmec_files = ["/home/erodrigu/Python/configs/wout_preciseQA.nc",
                    #   regcoil_run_folders[1] + "wout_downsampled_preciseQAs_target_0.50.nc",
                    #  regcoil_run_folders[2] + "wout_downsampled_preciseQAs_target_0.72.nc",
                        regcoil_run_folders[1] + "wout_downsampled_preciseQAs_target_0.30.nc",
                        regcoil_run_folders[2] + "wout_downsampled_preciseQAs_target_0.12.nc"
                    ]
    
    if s_values is None:
        s_values = [1.0, 
                    # 0.5, 
                    #0.72,
                    0.3, 0.12
                    ]
    
    a_minor = []
    for vmec_file in vmec_files:
        vs = load_vmec(vmec_file)
        a_minor.append(vs.Aminor_p)
    a_minor = np.array(a_minor)/a_minor_ref
    a_minor[a_minor == 0.0] = 1.0
    assert len(a_minor) == len(s_values), "Length of a_minor array must match length of s_values array."
    assert np.max(a_minor - np.sqrt(s_values)) < 0.05, "a_minor values do not match expected sqrt(s) scaling. Check the vmec files and the loading of a_minor."
    a_minor = np.sqrt(s_values)


    regcoil_results = {}
    for j, folder in enumerate(regcoil_run_folders):
        with open(folder + "coil_excursion_results.pkl", "rb") as f:
            data = pickle.load(f)
            print(data.keys())
            regcoil_results[folder] = {}
            print(data['distances'])
            regcoil_results[folder]["distances"] = data['distances']/a_minor_ref
            regcoil_results[folder]["max_excursions"] = data['max_excursions']
            regcoil_results[folder]["max_curvatures"] = data['max_curvatures']
            regcoil_results[folder]["max_geo_curvatures"] = data['max_geo_curvatures']
            regcoil_results[folder]["max_Bnormal_values"] = data['max_Bnormal_values']


    # Load VMEC results
    with open(vmec_coil_folder + "coil_excursion_results_vmec.pkl", "rb") as f:
        data_vmec = pickle.load(f)
    vmec_results = {}
    vmec_results["distances"] = np.sqrt(data_vmec["s_values"])
    vmec_results["max_excursions"] = data_vmec["max_excursions"]
    vmec_results["max_curvatures"] = data_vmec["max_curvatures"]
    vmec_results["max_geo_curvatures"] = data_vmec["max_geo_curvatures"]

    # Fit a quadratic curve to the max excursion data (ignoring the last point which is an outlier)
    coeffs = np.polyfit(vmec_results['distances'], vmec_results['max_excursions'], 2)
    # Fit a quadratic curve to the max curvature data (ignoring the last point which is an outlier)
    coeffs_curv = np.polyfit(vmec_results['distances'], vmec_results['max_curvatures'], 2)
    # Add ideal matching to others
    for j, (folder, results) in enumerate(regcoil_results.items()):
        results['ideal_max_excursions'] = np.polyval(coeffs, np.sqrt(s_values[j]))
        results['ideal_max_curvatures'] = np.polyval(coeffs_curv, np.sqrt(s_values[j]))

    ########
    # PLOT #
    ########
    plt.figure(figsize=(6, 4))
    # VMEC results first
    plt.plot(vmec_results['distances'], np.array(vmec_results['max_excursions'])/a_minor_ref, 'k', label='\\texttt{VMEC}')
    # Fit a quadratic curve to the max excursion data (ignoring the last point which is an outlier)
    coeffs = np.polyfit(vmec_results['distances'], np.array(vmec_results['max_excursions'])/a_minor_ref, 2)
    full_domain = np.linspace(0.7, 2.0, 100)
    fit_curve = np.polyval(coeffs, full_domain)
    plt.plot(full_domain, fit_curve, 'k:')
    # plt.axvline(1 + 0.005/a_minor_ref, color='silver', linestyle='-')

    # REGCOIL results
    for j, (folder, results) in enumerate(regcoil_results.items()):
        # Only plot if Bnormal below threshold
        valid_indices = np.where(np.array(results['max_Bnormal_values']) < 1e-3)[0]
        # Only keep monotonically increasing max excursion points
        if len(valid_indices) > 1:
            max_excursions = np.array(results['max_excursions'])
            valid_indices = valid_indices[np.where(np.diff(max_excursions[valid_indices]) > 0)[0]]
            valid_indices = np.append(valid_indices, valid_indices[-1] + 1)
        pos_vals = np.append([a_minor[j]], a_minor[j] + results['distances'])
        max_exc_vals = np.append([results['ideal_max_excursions']], results['max_excursions'])/a_minor_ref
        if j == 0:
            plt.scatter(pos_vals[valid_indices], max_exc_vals[valid_indices], c='k', marker='o', s = 10, label='\\texttt{REGCOIL}')
        else:
            plt.scatter(pos_vals[valid_indices], max_exc_vals[valid_indices], c='k', marker='o', s = 10)
            
    # plt.scatter(a_minor[0] + regcoil_results[regcoil_run_folders[0]]['distances'][0], regcoil_results[regcoil_run_folders[0]]['max_excursions'][0], c='g', marker='o', s = 10)


    # Plot ρ^2 scaling for reference
    plt.plot(full_domain, thry_scale*full_domain**2/a_minor_ref, 'k--')
    plt.text(0.8, thry_scale*2/5/a_minor_ref, r'$\propto \rho^2$', fontsize=15, color='k')

    # Initial points regcoil
    plt.scatter(a_minor[0], regcoil_results[regcoil_run_folders[0]]['ideal_max_excursions']/a_minor_ref, c='w', edgecolors='k', marker='s', s = 50, zorder = 10)
    plt.scatter(a_minor[1], regcoil_results[regcoil_run_folders[1]]['ideal_max_excursions']/a_minor_ref, c='w', edgecolors='k', marker='s', s = 50, zorder = 10)
    plt.scatter(a_minor[2], regcoil_results[regcoil_run_folders[2]]['ideal_max_excursions']/a_minor_ref, c='w', edgecolors='k', marker='s', s = 50, zorder = 10)

    # Final points regcoil: last not NaN point for each
    for j, (folder, results) in enumerate(regcoil_results.items()):
        distances = results['distances']
        max_excursions = results['max_excursions']
        valid_indices = np.where(~np.isnan(max_excursions))[0]
        print(f"Valid indices for {folder}: {len(valid_indices)}/{len(max_excursions)}")
        if len(valid_indices) > 0:
            last_valid_index = valid_indices[-1]
            plt.scatter(a_minor[j] + distances[last_valid_index], max_excursions[last_valid_index]/a_minor_ref, c='w', edgecolors='k', marker='^', s = 50, zorder = 10)

    plt.ylabel('Max Excursion')
    plt.yscale('log')
    # plt.xscale('log')
    plt.ylim(None, y_lim_top/a_minor_ref)
    plt.xlim(0.3, 2.)
    plt.xlabel('$$\\rho$$')
    plt.ylabel('$$\\Delta z/a$$')
    # Change x-tick labels
    plt.xticks([0.5, 1.0, 1.5, 2.0], [r'$0.5$', r'$1$', r'$1.5$', r'$2$'])
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.savefig(vmec_coil_folder + "excursion_distance_scan.pdf", dpi=150)


    # plt.figure(figsize=(6, 4))
    # # VMEC results first
    # plt.plot(vmec_results['distances'], vmec_results['max_curvatures'], 'k', label='\\texttt{VMEC}')
    # # Fit a quadratic curve to the max curvature data (ignoring the last point which is an outlier)
    # coeffs = np.polyfit(vmec_results['distances'], vmec_results['max_curvatures'], 2)

    # # REGCOIL results
    # for j, (folder, results) in enumerate(regcoil_results.items()):
    #     plt.scatter(np.append([a_minor[j]], a_minor[j] + results['distances']), \
    #              np.append([results['ideal_max_curvatures']], results['max_curvatures']), c='k', marker='o', s = 10)

    # # Plot 1/ρ scaling for reference
    # full_domain = np.linspace(0.35, 0.7, 100)
    # plt.plot(full_domain, 80/full_domain, 'k--')
    # plt.text(0.45, 210, r'$\propto 1/\rho$', fontsize=15, color='k')


    # plt.ylabel('Max Curvature')
    # plt.yscale('log')
    # # plt.ylim(None, 0.15)
    # plt.xlabel('$$d$$')
    # plt.xscale('log')
    # plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()

def get_minor_radius_from_vmec(vmec_file):
    vs = load_vmec(vmec_file)
    print("Check all attributes of vs: ", dir(vs))
    print(vs.Aminor_p)
    

