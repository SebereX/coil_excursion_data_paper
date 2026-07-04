from fileinput import filename
import os
import subprocess
import shutil
import numpy as np
from sys import stdout
from scipy.io import netcdf_file
from xarray import Dataset
from utils_surface_current import evaluate_on_grid, Struct, differentiate_spectral

_REGCOIL_EXECUTABLE = os.environ.get("REGCOIL_EXECUTABLE", "regcoil")

def solve_merkel_problem(vmec_file, plasma_coil_distance = 0.5, net_poloidal_current = 1.0, net_toroidal_current = 0.0, temp_dir = "./temp/", verbose = False, output_dir="./regcoil_results/", base_filename="temp", save = True, load = False, regcoil_omp_threads=2, **kwargs):
    """
    Given a VMEC equilibrium file and a desired plasma-coil distance, solve the Merkel problem using REGCOIL and return the resulting winding surface geometry and current potential.

    Parameters
    ----------
    vmec_file : str
        The name of the VMEC equilibrium file to be used as input for the REGCOIL run. 
    plasma_coil_distance : float
        The desired fraction of minor radius distance between the plasma and the coils.
    net_poloidal_current : float
        The net poloidal current in Amperes.
    net_toroidal_current : float
        The net toroidal current in Amperes.
    temp_dir : str
        The name of the temporary directory to be used for storing the REGCOIL input and output files.
    verbose: bool
        Whether to print verbose output during the REGCOIL run.
    save : bool
        Whether to save the REGCOIL input and output files.
    output_dir : str
        The directory where the REGCOIL input and output files will be saved.
    base_filename : str
        The base filename to use when saving the REGCOIL files. The input file will be saved as 'regcoil_in.{base_filename}' and the output file will be saved as 'regcoil_out.{base_filename}.nc'.
    **kwargs
        Additional keyword arguments to be passed to the REGCOIL input file creation function. 

    Returns
    -------
    data : dict
        A dictionary containing the relevant data:
        - 'mnmax_coil', 'xm_coil', 'xn_coil', 'rmnc_coil', 'zmns_coil': the Fourier coefficients of the winding surface geometry
        - 'mnmax_potential', 'xm_potential', 'xn_potential', 'single_valued_current_potential_mn': the Fourier coefficients of the single-valued current potential on the plasma surface
        - 'net_poloidal_current_Amperes', 'net_toroidal_current_Amperes': the net poloidal and toroidal currents in Amperes
        - 'chi2_B', 'max_Bnormal': error in construction of the field.
    """
    # Check tempirary directory exists, if not create it
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    if not load:
        ###########################
        # MAKE REGCOIL INPUT FILE #
        ###########################
        # Create the REGCOIL input file
        vmec_file_abs = os.path.abspath(vmec_file)
        input_file_name = os.path.join(temp_dir, "regcoil_in.temp")
        input_parameters = {
                            "separation": plasma_coil_distance,
                            "net_poloidal_current_Amperes": net_poloidal_current,
                            "net_toroidal_current_Amperes": net_toroidal_current,
                            "wout_filename": f"'{vmec_file_abs}'",
                            "nlambda": 1,
        }

        input_parameters.update(kwargs)
        create_regcoil_in_file(input_file_name, input_parameters)

        ###############
        # RUN REGCOIL #
        ###############
        if verbose:
            print("Running REGCOIL...")
        run_log = os.path.join(temp_dir, "regcoil_run_temp")
        run_code = run_regcoil(input_file_name, run_log = run_log, verbose = verbose, omp_threads=regcoil_omp_threads)
        if verbose:
            print("REGCOIL run complete.")
        
        if save:
            save_regcoil_files(temp_folder = temp_dir, output_dir=output_dir, base_filename=base_filename, not_out = (run_code != 0))
            if verbose:
                print(f"REGCOIL input and output files saved to '{output_dir}' with base filename '{base_filename}'.")
        
        if run_code != 0:
            raise RuntimeError(f"REGCOIL run failed with return code {run_code}. Check the log file '{run_log}.log' for details.")


    ###############
    # READ OUTPUT #
    ###############
    data_filt = read_output_dict(temp_dir, output_dir, base_filename, plasma_coil_distance, verbose = verbose, load = load)

    return data_filt

def read_output_dict(temp_dir, output_dir, base_filename, plasma_coil_distance, verbose = False, load = False):
    ###############
    # READ OUTPUT #
    ###############
    output_file_name = os.path.join(temp_dir, "regcoil_out.temp.nc") if not load else os.path.join(output_dir, f"regcoil_out.{base_filename}.nc")
    data = read_regcoil_output(output_file_name, verbose = verbose)

    # Select lambda = 0.0
    if 'lambda' in data and len(data['lambda']) > 1:
        lambda_index = np.argmin(np.abs(data['lambda'] - 0.0))
    else:
        lambda_index = 0

    # Select only relevant data to return
    relevant_keys_nl = ['mnmax_coil', 'xm_coil', 'xn_coil', 
                     'mnmax_potential', 'xm_potential', 'xn_potential', 
                        'net_poloidal_current_Amperes', 'net_toroidal_current_Amperes',
                        ]
    relevant_keys_l = ['rmnc_coil', 'zmns_coil',
                       'single_valued_current_potential_mn',
                       'chi2_B', 'max_Bnormal'
                       ]

    data_filt = {key: data[key] for key in relevant_keys_nl if key in data}
    data_filt.update({key: np.squeeze(data[key][lambda_index, ...]) for key in relevant_keys_l if key in data})
    data_filt["distances"] = plasma_coil_distance

    return data_filt

def make_current_density_structure(data):
    """
    Make structure to interface with coil cutting and property measuring.

    Parameters
    ----------
    data : dict
        Data dictionary as results from solve_merkel_problem, containing at least the following keys:   
        - 'mnmax_coil', 'xm_coil', 'xn_coil', 'rmnc_coil', 'zmns_coil': the Fourier coefficients of the winding surface geometry
        - 'mnmax_potential', 'xm_potential', 'xn_potential', 'single_valued_current_potential_mn': the Fourier coefficients of the single-valued current potential on the plasma surface
        - 'net_poloidal_current_Amperes', 'net_toroidal_current_Amperes': the net poloidal and toroidal currents in Amperes
    Returns
    -------
    current_density_struct : Struct
        A structure containing the current density information, with the following attributes:
        - 'xm', 'xn', 'mnmax', 'rmnc', 'zmns' : for winding surface
        - 'xm_nyq', 'xn_nyq', 'mnmax_nyq', 'bsubumnc', 'bsubvmnc': for following Phi contours
        - 'phimns', 'G_current', 'I_current': for current potential and net currents
    """
    # Check that necessary data is present
    required_keys = ['mnmax_coil', 'xm_coil', 'xn_coil', 'rmnc_coil', 'zmns_coil',
                     'mnmax_potential', 'xm_potential', 'xn_potential', 'single_valued_current_potential_mn', 'net_poloidal_current_Amperes', 'net_toroidal_current_Amperes']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Data dictionary must contain key '{key}' to make current density structure.")

    # Make structure
    current_density_struct = Struct()
    current_density_struct.contents = ['xm', 'xn', 'mnmax', 'rmnc', 'zmns', 'xm_nyq', 'xn_nyq', 'mnmax_nyq', 'bsubumnc', 'bsubvmnc', 'phimns', 'G_current', 'I_current']
    current_density_struct.xm = data['xm_coil']
    current_density_struct.xn = data['xn_coil']
    current_density_struct.mnmax = data['mnmax_coil']
    current_density_struct.rmnc = data['rmnc_coil']
    current_density_struct.zmns = data['zmns_coil']

    # Construct bsubu and bsubv as derivatives of current potential
    # Get m=0, n=0 index
    m0n0_index = np.where((data['xm_potential'] == 0) & (data['xn_potential'] == 0))
    if len(m0n0_index[0]) == 0:
        # Insert m=0, n=0 mode if not present
        current_density_struct.xm_nyq = np.insert(data['xm_potential'], 0, 0)
        current_density_struct.xn_nyq = np.insert(data['xn_potential'], 0, 0)
        current_density_struct.mnmax_nyq = data['mnmax_potential'] + 1
        current_density_struct.phimns = np.insert(data['single_valued_current_potential_mn'], 0, 0)
        m0n0_index = (0,)  # Now at index 0
    else:
        current_density_struct.xm_nyq = data['xm_potential']
        current_density_struct.xn_nyq = data['xn_potential']
        current_density_struct.mnmax_nyq = data['mnmax_potential']
        current_density_struct.phimns = data['single_valued_current_potential_mn']

    # Get current
    G_current = data['net_poloidal_current_Amperes']
    I_current = data['net_toroidal_current_Amperes']

    # Compute the partial derivatives of Phi
    bsubu = differentiate_spectral(current_density_struct.phimns, current_density_struct.xm_nyq, current_density_struct.xn_nyq, s_or_c = "s", m_or_n = "m")    
    bsubu[m0n0_index] = I_current/2/np.pi
    bsubv = differentiate_spectral(current_density_struct.phimns, current_density_struct.xm_nyq, current_density_struct.xn_nyq, s_or_c = "s", m_or_n = "n")
    bsubv[m0n0_index] = G_current/2/np.pi
    current_density_struct.bsubumnc = bsubu
    current_density_struct.bsubvmnc = bsubv
    current_density_struct.G_current = G_current
    current_density_struct.I_current = I_current

    # mu0 = 4*np.pi*1e-7
    # current_density_struct.G_Boozer = G_current * mu0 / (2*np.pi)
    # current_density_struct.I_Boozer = I_current * mu0 / (2*np.pi)

    return current_density_struct

def run_regcoil(input_file_name, run_log = "regcoil_run_", verbose = False, omp_threads=2):
    """
    Launch regcoil run onto the console, record output to terminal onto a file, and return name of output files.

    Parameters
    ----------
    input_file_name : str
        The name of the input file to be used for the REGCOIL run. This file should be in the current working directory.

    Returns
    -------
    output_file_names : list of str
        A list of the names of the output files generated by REGCOIL. The specific output files generated will depend on the contents of the input file and the configuration of REGCOIL, but typically include files with names starting with 'regcoil_out.'.

    """
    # Check input file exists
    if not os.path.isfile(input_file_name):
        raise FileNotFoundError(f"Input file '{input_file_name}' does not exist.")

    # Check input file is *regcoil_in.*
    if "regcoil_in." not in os.path.basename(input_file_name):
        raise ValueError(f"Input file '{input_file_name}' does not have the expected format 'regcoil_in.*'.")

    # Get directory and basename
    input_dir = os.path.dirname(os.path.abspath(input_file_name))
    if verbose:
        print("Input directory: ", input_dir)
    input_basename = os.path.basename(input_file_name)

    # Construct the command to run REGCOIL (pass only the basename)
    submitCommand2 = _REGCOIL_EXECUTABLE + " " + input_basename
    submitCommandSplit = submitCommand2.split()
    if verbose:
        print("About to submit the following command (in dir):", input_dir)
        print("  ", submitCommandSplit)

    # Flush everything printed into a file
    log_file_name = run_log + f"{input_basename}.log"
    try:
        env = os.environ.copy()
        if omp_threads is not None:
            env["OMP_NUM_THREADS"] = str(omp_threads)
        with open(log_file_name, "w") as logfile:
            process = subprocess.Popen(
                submitCommandSplit,
                cwd=input_dir,  # Run in directory
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )
            for line in process.stdout:
                if verbose:
                    print(line, end="")
                logfile.write(line)
                logfile.flush()
            process.stdout.close()
            returncode = process.wait()
            if returncode != 0:
                print(f"REGCOIL exited with return code {returncode}")
    except Exception as e:
        print("An error occurred when attempting to launch REGCOIL.")
        returncode = 0
        raise e

    if verbose:
        print(" ")
        print("REGCOIL execution complete. About to run tests on output.")
    stdout.flush()

    # Check if output file was generated
    output_file_name = os.path.join(input_dir, "regcoil_out.temp.nc")
    if not os.path.isfile(output_file_name):
        raise FileNotFoundError(f"Expected output file '{output_file_name}' was not generated by REGCOIL.")
    # Check also within log file "You can run regcoilPlot "
    if not os.path.isfile(log_file_name):
        raise FileNotFoundError(f"Expected log file '{log_file_name}' was not generated by REGCOIL.")
    
    # Check if STOP appears in the log file, which indicates a failure
    with open(log_file_name, "r") as logfile:
        log_contents = logfile.read()
        if "STOP" in log_contents:
            returncode = -1

    return returncode

def create_regcoil_in_file(input_file_name, input_parameters):
    """
    Create a REGCOIL input file with the specified name and parameters.

    Parameters
    ----------
    input_file_name : str
        The name of the input file to be created. This file will be created in the current working directory.
    input_parameters : dict
        A dictionary containing the parameters to be written to the REGCOIL input file. The keys of the dictionary should correspond to the parameter names expected by REGCOIL, and the values should be the corresponding parameter values.

    Returns
    -------
    None
        This function does not return anything. It creates a file in the current working directory with the specified name and contents based on the provided parameters.

    """
    # Check that input file name is valid
    if "regcoil_in." not in input_file_name:
        raise ValueError(f"Input file name '{input_file_name}' does not have the expected format 'regcoil_in.*'.")

    # Create the content of the REGCOIL input file based on the provided parameters

    # --- Full parameter set and dependencies from manual ---
    # Defaults (update as needed for your use case)
    defaults = {
        # General
        "general_option": 1,
        "regularization_term_option": "'chi2_K'",
        "symmetry_option": 1,
        "save_level": 3,
        "load_bnorm": ".false.",
        "net_poloidal_current_Amperes": 1.0,
        "net_toroidal_current_Amperes": 0.0,
        # Resolution
        "ntheta_plasma": 64,
        "ntheta_coil": 64,
        "nzeta_plasma": 64,
        "nzeta_coil": 64,
        "mpol_potential": 12,
        "ntor_potential": 12,
        # Geometry plasma
        "geometry_option_plasma": 2,
        # Geometry coil
        "geometry_option_coil": 2,
        # Regularization
        "nlambda": 4,
        "lambda_max": 1.0e-13,
        "lambda_min": 1.0e-19,
    }
    # Merge user parameters
    params = defaults.copy()
    params.update(input_parameters)

    # Helper: Only include parameter if relevant
    def include_param(key):
        go = params.get("general_option", 1)
        gop = params.get("geometry_option_plasma", 2)
        goc = params.get("geometry_option_coil", 2)
        load_bnorm = params.get("load_bnorm", ".false.")
        sens_opt = params.get("sensitivity_option", 1)
        target_opt = params.get("target_option", None)
        # General_option dependencies
        if key in ["lambda_min", "lambda_max"] and go != 1:
            return False
        if key == "Nlambda" and go not in [1, 4, 5]:
            return False
        if key == "target_option" and go not in [4, 5]:
            return False
        if key == "target_value" and go not in [4, 5]:
            return False
        if key == "lambda_search_tolerance" and go not in [4, 5]:
            return False
        if key == "nescout_filename" and go != 2:
            return False
        if key == "load_bnorm" and go not in [1, 3]:
            return False
        if key == "bnorm_filename" and (go not in [1, 3] or load_bnorm != ".true."):
            return False
        # Geometry_option_plasma dependencies
        if key in ["R0_plasma", "a_plasma", "nfp_imposed"] and gop not in [0, 1]:
            return False
        if key == "wout_filename" and gop not in [2, 3, 4]:
            return False
        if key in ["efit_filename", "efit_psiN", "efit_num_modes"] and gop != 5:
            return False
        if key == "shape_filename_plasma" and gop not in [6, 7]:
            return False
        if key in ["mpol_transform_refinement", "ntor_transform_refinement"] and gop != 4:
            return False
        # Geometry_option_coil dependencies
        if key in ["R0_coil"] and goc != 1:
            return False
        if key in ["a_coil"] and goc not in [0, 1]:
            return False
        if key == "separation" and goc != 2:
            return False
        if key == "nescin_filename" and goc not in [2, 3]:
            return False
        if key in ["mpol_coil_filter", "ntor_coil_filter"] and goc not in [2, 3, 4]:
            return False
        # Regularization dependencies
        if key == "target_option_p" and not (target_opt in ["'lp_norm_K'", "'max_K_lse'"]):
            return False
        # Sensitivity dependencies
        if key in ["sensitivity_option"] and sens_opt <= 1:
            return False
        if key in ["fixed_norm_sensitivity_option", "nmax_sensitivity", "mmax_sensitivity", "coil_plasma_dist_lse_p"] and sens_opt <= 1:
            return False
        return True

    # String parameters that must be quoted
    str_parameters = [
        "regularization_term_option", "wout_filename", "bnorm_filename", "nescout_filename", "efit_filename", "shape_filename_plasma", "nescin_filename", "target_option"
    ]
    # Logical parameters
    logical_parameters = ["load_bnorm", "fixed_norm_sensitivity_option"]

    # Validate string quoting and logicals
    for key, value in params.items():
        if key in str_parameters and not (isinstance(value, str) and value.startswith("'") and value.endswith("'")):
            raise ValueError(f"String parameter '{key}' should be enclosed in single quotes (').")
        if key in logical_parameters and not (str(value).lower() in [".true.", ".false.",".t.", ".f."]):
            raise ValueError(f"Logical parameter '{key}' should be .true. or .false.")

    # Write file
    try:
        with open(input_file_name, "w") as input_file:
            input_file.write("! This is a REGCOIL input file generated by regcoil_utils.py\n")
            input_file.write("! For documentation of the input parameters and their default values, see the REGCOIL manual.\n\n")
            input_file.write("&regcoil_nml\n")
            for key in params:
                if include_param(key):
                    input_file.write(f"  {key} = {params[key]}\n")
            input_file.write("/\n")
    except Exception as e:
        print("An error occurred when attempting to create the REGCOIL input file.")
        raise e

    return input_file_name

def read_regcoil_output(output_file_name, verbose = False):
    """
    Read the output from a REGCOIL run and return the relevant data as numpy arrays.

    Parameters
    ----------
    output_file_name : str
        The name of the output file generated by REGCOIL. This file should be in the current working directory and should be a netCDF file containing the results of a REGCOIL run.

    Returns
    -------
    data : dict
        A dictionary containing the relevant data from the REGCOIL output file. The specific keys and values in the dictionary will depend on the contents of the output file, but typically include arrays for the plasma surface geometry (e.g., 'r_plasma', 'z_plasma'), the coil geometry (e.g., 'r_coil', 'z_coil'), the current potential on the plasma surface (e.g., 'lambda'), and the normal magnetic field on the plasma surface (e.g., 'Bnormal_total'). The dictionary may also include other relevant data from the REGCOIL output file, such as the chi-squared values for the fit, the coil currents, and any other variables that were included in the output file.
    """
    # Check file exists
    if not os.path.isfile(output_file_name):
        raise FileNotFoundError(f"Output REGCOIL file '{output_file_name}' does not exist.")

    # Load netcdf file
    f = netcdf_file(output_file_name, 'r', mmap=False)
    print("Reading REGCOIL output from file: ", output_file_name)

    data = {}
    # Read all variables present in the file
    def var_theta_zeta_shuffle(var):
        """
        Read data and shuffle theta and zeta dimensions if both are present, to ensure theta is the first dimension and zeta is the second dimension. This is needed because REGCOIL outputs FORTRAN-ordered arrays.

        Parameters
        ----------
        var : netCDF variable
            A variable from the netCDF file.

        Returns
        -------
        val : numpy array
            The data from the variable, with theta and zeta dimensions shuffled if both are present.
        """
        # If both zeta and theta as part of the dimension
        dims = var.dimensions
        is_thetas = ["theta" in dim for dim in dims]
        is_theta = any(is_thetas)
        is_zetas = ["zeta" in dim for dim in dims]
        is_zeta = any(is_zetas)
        if is_theta and is_zeta:
            # Place the theta dim before the zeta dim
            dim_order = np.arange(len(dims))
            ind_theta = np.where(is_thetas)[0][0]
            ind_zeta = np.where(is_zetas)[0][0]
            dim_order[[ind_theta, ind_zeta]] = dim_order[[ind_zeta, ind_theta]]
            val = np.transpose(var[()], axes=dim_order)
        else:
            val = var[()]
        return val

    for key in f.variables:
        try:
            # Read the variable data into a numpy array
            data[key] = var_theta_zeta_shuffle(f.variables[key])
        except Exception as e:
            print(f"Warning: Could not read variable '{key}': {e}")
            data[key] = None

    f.close()

    # If verbose: print all keys and shapes
    if verbose:
        print("Data keys and shapes:")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape {value.shape}")
            else:
                print(f"  {key}: {value}")

    # Backward compatibility for lambda/alpha and K/J naming
    if 'lambda' not in data and 'alpha' in data:
        data['lambda'] = data['alpha']
    if 'nlambda' not in data and 'nalpha' in data:
        data['nlambda'] = data['nalpha']
    if 'chi2_K' not in data and 'chi2_J' in data:
        data['chi2_K'] = data['chi2_J']
    if 'K2' not in data and 'J2' in data:
        data['K2'] = data['J2']

    # Patch for all-zero lambda array
    if 'lambda' in data and np.max(np.abs(data['lambda'])) < 1.0e-200:
        print("lambda array appears to be all 0. Changing it to all 1 to avoid a python error.")
        data['lambda'] += 1

    # Sort only variables known to be indexed by lambda, if lambda exists and has more than one value
    if 'lambda' in data and len(data['lambda']) > 1:
        lambdas = data['lambda']
        if lambdas.ndim == 1 and len(lambdas) > 1:
            permutation = np.argsort(lambdas)
            # Sort lambda itself
            data['lambda'] = lambdas[permutation]
            # List of variable names that are lambda-indexed (from user output)
            lambda_indexed_vars = [
                'chi2_B', 'chi2_K', 'chi2_Laplace_Beltrami', 'max_Bnormal', 'max_K',
                'single_valued_current_potential_mn', 'single_valued_current_potential_thetazeta',
                'current_potential', 'Bnormal_total', 'K2', 'Laplace_Beltrami2'
            ]
            for key in lambda_indexed_vars:
                if key in data and isinstance(data[key], np.ndarray) and data[key].shape[0] == len(lambdas):
                    data[key] = data[key][permutation, ...]
            # Patch: set last lambda to np.inf if extremely large
            if data['lambda'][-1] > 1.0e199:
                data['lambda'][-1] = np.inf

    # Transpose theta/zeta indexes of arrays (FORTRAN )

    return data

def save_regcoil_files(temp_folder = "./temp_test/", output_dir="./regcoil_results/", base_filename="temp", not_out = False):
    """
    Save the REGCOIL input and output files from a run in a specified directory with a specified base filename to a specified location.

    Parameters
    ----------
    temp_folder : str
        The name of the temporary directory where the REGCOIL input and output files are currently stored.
    output_dir : str
        The name of the directory where the REGCOIL files should be saved. If the directory does not exist, it will be created.
    base_filename : str
        The base filename to use when saving the REGCOIL files. The input file will be saved as 'regcoil_in.{base_filename}' and the output file will be saved as 'regcoil_out.{base_filename}.nc'.
    not_out : bool
        If True, indicates that the REGCOIL run did not produce an output file (e.g., due to a failure), so only the input file and log file will be saved, and the function will not attempt to save the output file.

    Returns
    -------
    None
        This function does not return anything. It saves the REGCOIL input and output files to the specified directory with the specified filenames.

    """
    # Check temp_folder exists
    if not os.path.exists(temp_folder):
        raise FileNotFoundError(f"Temporary folder '{temp_folder}' does not exist.")

    # Check output_dir exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define source and destination file paths
    input_src = os.path.join(temp_folder, "regcoil_in.temp")
    log_src = os.path.join(temp_folder, "regcoil_run_tempregcoil_in.temp.log")
    input_dst = os.path.join(output_dir, f"regcoil_in.{base_filename}")
    output_log_dst = os.path.join(output_dir, f"regcoil_run_{base_filename}.log")

    if not not_out:
        output_src = os.path.join(temp_folder, "regcoil_out.temp.nc")
        output_dst = os.path.join(output_dir, f"regcoil_out.{base_filename}.nc")

    # Check source files exist
    if not os.path.isfile(input_src):
        raise FileNotFoundError(f"Expected input file '{input_src}' does not exist.")
    if not not_out and not os.path.isfile(output_src):
        raise FileNotFoundError(f"Expected output file '{output_src}' does not exist.")
    if not os.path.isfile(log_src):
        raise FileNotFoundError(f"Expected log file '{log_src}' does not exist.")


    # Copy files to destination
    try:
        shutil.copy(input_src, input_dst)
        if not not_out:
            shutil.copy(output_src, output_dst)
        shutil.copy(log_src, output_log_dst)
        print(f"REGCOIL files saved to '{output_dir}' with base filename '{base_filename}'.")
    except Exception as e:
        print("An error occurred when attempting to save the REGCOIL files.")
        raise e

#########
# TESTS #
#########

def modify_surface_nescin(filename, xn, xm, rmnc, zmns):
    """
    Modify the surface in a nescin file with new Fourier coefficients.

    Parameters
    ----------
    filename : str
        The name of the nescin file to be modified. This file should be in the current working directory and should be a netCDF file containing the surface geometry in Fourier coefficients.
    xn : array-like
        The new toroidal mode numbers for the surface geometry.
    xm : array-like
        The new poloidal mode numbers for the surface geometry.
    rmnc : 2D array-like
        The new Fourier coefficients for the R component of the surface geometry, indexed by (m, n).
    zmns : 2D array-like
        The new Fourier coefficients for the Z component of the surface geometry, indexed by (m, n).

    Returns
    -------
    None
        This function does not return anything. It modifies the specified nescin file in place with the new Fourier coefficients for the surface geometry.

    """
    # Check file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"NESCIN file '{filename}' does not exist.")

#     ------ Plasma information from VMEC ----
# np     iota_edge       phip_edge       curpol
#      2  0.000000000000E+00  0.000000000000E+00  4.007279974966E+01
# 
# ------ Current Surface: Coil-Plasma separation =   1.700000000000E-01 -----
# Number of fourier modes in table
#         1201
# Table of fourier coefficients
# m,n,crc2,czs2,crs2,czc2
#       0     0  9.998240737637E-01  0.000000000000E+00  0.000000000000E+00  0.000000000000E+00
#       0    -1  1.795266750056E-01 -1.497112581210E-01  0.000000000000E+00  0.000000000000E+00

    # Load file with structure above


    # # Read the xm, xn, rmnc, zmns, rmns, zmnc variables
    # xm = f.variables['xm_coil'][:]
    # xn = f.variables['xn_coil'][:]

    # # Check that necessary variables are present
    # required_vars = ['xm_coil', 'xn_coil', 'rmnc_coil', 'zmns_coil']
    # for var in required_vars:
    #     if var not in f.variables:
    #         raise ValueError(f"Variable '{var}' not found in NESCIN file '{filename}'.")

    # # Modify variables with new values
    # f.variables['xm_coil'][:] = xn
    # f.variables['xn_coil'][:] = xm
    # f.variables['rmnc_coil'][:] = rmnc
    # f.variables['zmns_coil'][:] = zmns

    # f.close()


# ----------------------
# ASCII surface file IO
# ----------------------
import re
def load_nescin_ascii_surface(filename):
    """
    Load a nescin ASCII surface file with the structure:
    ------ Plasma information from VMEC ----
    np     iota_edge       phip_edge       curpol
         2  0.000000000000E+00  0.000000000000E+00  4.007279974966E+01
    ------ Current Surface: Coil-Plasma separation =   1.700000000000E-01 -----
    Number of fourier modes in table
            1201
    Table of fourier coefficients
    m,n,crc2,czs2,crs2,czc2
    ...
    Returns:
        meta: dict with plasma info and separation
        coeffs: dict with numpy arrays for m, n, crc2, czs2, crs2, czc2
    """
    import numpy as np
    meta = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Parse plasma info
    for i, line in enumerate(lines):
        if line.strip().startswith('------ Plasma information from VMEC'):
            meta['plasma_info_header'] = lines[i].strip()
            meta['plasma_info_labels'] = lines[i+1].strip()
            meta['plasma_info_values'] = lines[i+2].strip()
            vals = lines[i+2].split()
            if len(vals) == 4:
                meta['np'] = int(vals[0])
                meta['iota_edge'] = float(vals[1])
                meta['phip_edge'] = float(vals[2])
                meta['curpol'] = float(vals[3])
        if line.strip().startswith('------ Current Surface: Coil-Plasma separation'):
            match = re.search(r'separation =\s*([\d.Ee+-]+)', line)
            if match:
                meta['coil_plasma_separation'] = float(match.group(1))
        if line.strip().startswith('Number of fourier modes in table'):
            meta['n_modes'] = int(lines[i+1].strip())
        if line.strip().startswith('Table of fourier coefficients'):
            header = lines[i+1].strip().split(',')
            meta['coeff_header'] = header
            coeff_start = i+2
            break
    # Read coefficient table
    coeff_data = []
    for line in lines[coeff_start:]:
        if not line.strip() or line.strip().startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) != 6:
            break
        coeff_data.append([float(x) for x in parts])
    arr = np.array(coeff_data)
    coeffs = dict(zip(meta['coeff_header'], arr.T))
    return meta, coeffs

def save_nescin_ascii_surface(filename, meta, coeffs):
    """
    Save the nescin ASCII surface file with updated coefficients.
    Args:
        filename: output file path
        meta: dict from load_nescin_ascii_surface
        coeffs: dict with keys matching meta['coeff_header'] and numpy arrays
    """
    n_modes = len(coeffs[meta['coeff_header'][0]])
    with open(filename, 'w') as f:
        # Write plasma info
        f.write(f"{meta.get('plasma_info_header', '------ Plasma information from VMEC ----')}\n")
        f.write(f"{meta.get('plasma_info_labels', 'np     iota_edge       phip_edge       curpol')}\n")
        f.write(f"{meta.get('plasma_info_values', '')}\n")
        f.write(f"\n------ Current Surface: Coil-Plasma separation =   {meta.get('coil_plasma_separation', 0.0):.12E} -----\n")
        f.write(f"Number of fourier modes in table\n")
        f.write(f"{n_modes:9d}\n")
        f.write(f"Table of fourier coefficients\n")
        f.write(",".join(meta['coeff_header']) + "\n")
        # Write coefficients with REGCOIL-style formatting
        for row in zip(*(coeffs[k] for k in meta['coeff_header'])):
            # m and n: right-aligned, 7-wide, at least 2 spaces between
            m = int(row[0])
            n = int(row[1])
            floats = row[2:]
            line = f"{m:7d}  {n:7d}"
            for v in floats:
                line += f"  {v: .12E}"
            f.write(line + "\n")
        # Ensure file ends with a newline
        f.flush()
    return filename

def comparison_anal_regcoil(
    vmec_file=None,
    regcoil_output_file=None,
    output_file="regcoil_phi_comparison_simple_true.png",
    show=True,
):
    """
    Test the full solve_merkel_problem function
    """
    #############
    # LOAD VMEC #
    #############
    from utils_surface_current import load_vmec, vmec_compute_geometry
    if vmec_file is None:
        vmec_file = os.environ.get("VMEC_WOUT_FILE")
    if vmec_file is None:
        raise ValueError("vmec_file must be provided or VMEC_WOUT_FILE must be set.")

    vs = load_vmec(vmec_file)
    s = [1.0]
    theta_c = np.linspace(0, 2 * np.pi, 200)
    phi_c = np.linspace(0, 2 * np.pi, 300)
    results = vmec_compute_geometry(vs, s, theta_c, phi_c)
    mu0 = 4*np.pi*1e-7
    G_current = results.G_Boozer[0] * 2*np.pi / mu0

    # Find Phi on the grid analytic
    from utils_surface_current import compute_Phi_values
    Phi_analytic_tilde = compute_Phi_values(results, theta_c, phi_c, secular = False); Phi_analytic_tilde = np.squeeze(Phi_analytic_tilde)
    Phi_analytic = compute_Phi_values(results, theta_c, phi_c, secular = True); Phi_analytic = np.squeeze(Phi_analytic)

    if regcoil_output_file is None:
        raise ValueError("regcoil_output_file must be provided.")

    data_raw = read_regcoil_output(regcoil_output_file, verbose=False)
    if 'lambda' in data_raw and len(data_raw['lambda']) > 1:
        lambda_index = np.argmin(np.abs(data_raw['lambda'] - 0.0))
    else:
        lambda_index = 0

    relevant_keys_nl = [
        'mnmax_coil', 'xm_coil', 'xn_coil',
        'mnmax_potential', 'xm_potential', 'xn_potential',
        'net_poloidal_current_Amperes', 'net_toroidal_current_Amperes',
    ]
    relevant_keys_l = [
        'rmnc_coil', 'zmns_coil',
        'single_valued_current_potential_mn',
        'chi2_B', 'max_Bnormal',
    ]
    data = {key: data_raw[key] for key in relevant_keys_nl if key in data_raw}
    data.update({key: np.squeeze(data_raw[key][lambda_index, ...]) for key in relevant_keys_l if key in data_raw})

    print("Goodness of fit: chi2_B = ", data['chi2_B'], ", max_Bnormal = ", data['max_Bnormal'])
    
    ############
    # PLOT PHI #
    ############  
    Phi = evaluate_on_grid(data['single_valued_current_potential_mn'], data['xm_potential'], data['xn_potential'], theta_c, phi_c, 's')
    Phi_max = np.max(np.abs(Phi))

    # Plot
    import matplotlib.pyplot as plt
    Theta_c, Phi_c = np.meshgrid(theta_c, phi_c, indexing='ij')

    # plt.savefig("regcoil_phi_comparison.png", dpi=300)
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.2]}, figsize=(11, 5))
    plt.sca(axs[0])
    plt.pcolormesh(theta_c/(2*np.pi), phi_c/(2*np.pi), Phi.T / Phi_max, shading='auto', cmap='RdYlBu')
    # plt.colorbar()
    plt.xlabel("$$\\theta/(2\\pi)$$")
    plt.ylabel("$$\\phi/(2\\pi)$$")
    plt.title("$$ \\tilde{\\Phi}_\\texttt{REGCOIL} / \\tilde{\\Phi}_{\\texttt{REGCOIL}, \\mathrm{max}} $$", fontsize=20)
    plt.sca(axs[1])
    Phi_anal_max = np.max(np.abs(Phi_analytic_tilde))
    plt.pcolormesh(theta_c/(2*np.pi), phi_c/(2*np.pi), Phi_analytic_tilde.T / Phi_anal_max, shading='auto', cmap='RdYlBu')
    plt.colorbar()
    plt.xlabel("$$\\theta/(2\\pi)$$")
    plt.ylabel("$$\\phi/(2\\pi)$$")
    plt.title("$$ \\tilde{\\Phi}_\\mathrm{an} / \\tilde{\\Phi}_{\\mathrm{an}, \\mathrm{max}} $$", fontsize=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


