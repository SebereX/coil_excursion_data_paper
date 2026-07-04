import time
import copy
import os
from simsopt.mhd import Vmec, Boozer
from simsopt.mhd.vmec_diagnostics import Struct, vmec_splines
from scipy.integrate import solve_ivp
from numba import njit
import matplotlib.pyplot as plt
try:
    import stell_tools
except ImportError:
    stell_tools = None
import numpy as np
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from tqdm import tqdm

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Serif'], 'size': 20})
rc('text', usetex=True)


def vmec_compute_geometry(vs, s, theta, phi, full_output = True):
    """
    Compute geometric quantities of interest from a vmec configuration. 

    Parameters
    ----------
    vs : Vmec or vmec_splines
        The input vmec configuration, either as a Vmec object or as vmec_splines.
    s : float or array
        The normalized toroidal flux values at which to evaluate the geometry. Can be a single float or an array of floats.
    theta : float or array
        The theta values on which to evaluate the geometry. Can be a single float or an array of floats. If full_output is False, then this argument is ignored.
    phi : float or array
        The phi values on which to evaluate the geometry. Can be a single float or an array of floats. If full_output is False, then this argument is ignored.
    full_output : bool
        Whether to evaluate the full geometry on a (theta, phi) grid. If False, then only the quantities that depend on s are evaluated and returned. Default is True.
    
    Returns
    -------
    results : Struct
        A structure containing the geometric quantities of interest, including:
        - ns: the number of s values
        - s: the s values at which the geometry was evaluated
        - iota: the rotational transform evaluated at the given s values
        - d_iota_d_s: the derivative of the rotational transform with respect to s, evaluated at the given s values
        - xm, xn: the m and n values corresponding to the Fourier modes in the R and Z Fourier expansions
        - rmnc, zmns: the Fourier coefficients of the R and Z Fourier expansions, evaluated at the given s values
        - xm_nyq, xn_nyq: the m and n values corresponding to the Fourier modes in the B field covariant components
        - bsubumnc, bsubvmnc: the Fourier coefficients of the covariant components of the magnetic field, evaluated at the given s values
        - I_Boozer, G_Boozer: the Boozer currents, evaluated at the given s values
        - edge_toroidal_flux_over_2pi: the toroidal flux at the edge divided by 2*pi, evaluated at the given s values

        If full_output is True, then the following additional quantities are also returned:
        - R, Z: the R and Z coordinates of the surface evaluated on the (theta, phi) grid
        - B_sub_theta_vmec, B_sub_phi: the covariant components of the magnetic field evaluated on the (theta, phi) grid
    """
    ########################
    # GET SPLINE FROM VMEC #
    ########################
    # If given a Vmec object, convert it to vmec_splines:
    if isinstance(vs, Vmec):
        vs = vmec_splines(vs)

    ######################
    # EVALUATE DATA ON s #
    ######################
    # Make sure s is an array:
    try:
        ns = len(s)
    except:
        s = [s]
    s = np.array(s)
    ns = len(s)

    # Shorthand
    mnmax = vs.mnmax
    xm = vs.xm
    xn = vs.xn
    mnmax_nyq = vs.mnmax_nyq
    xm_nyq = vs.xm_nyq
    xn_nyq = vs.xn_nyq

    # Now that we have an s grid, evaluate everything on that grid:
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)

    # Evaluate the R and Z Fourier coefficients
    rmnc = np.zeros((ns, mnmax))
    zmns = np.zeros((ns, mnmax))
    lmns = np.zeros((ns, mnmax))
    d_rmnc_d_s = np.zeros((ns, mnmax))
    d_zmns_d_s = np.zeros((ns, mnmax))
    d_lmns_d_s = np.zeros((ns, mnmax))
    for jmn in range(mnmax):
        rmnc[:, jmn] = vs.rmnc[jmn](s)
        zmns[:, jmn] = vs.zmns[jmn](s)
        lmns[:, jmn] = vs.lmns[jmn](s)
        d_rmnc_d_s[:, jmn] = vs.d_rmnc_d_s[jmn](s)
        d_zmns_d_s[:, jmn] = vs.d_zmns_d_s[jmn](s)
        d_lmns_d_s[:, jmn] = vs.d_lmns_d_s[jmn](s)

    # Evaluate the covariant components of the magnetic field
    bsubumnc = np.zeros((ns, mnmax_nyq))
    bsubvmnc = np.zeros((ns, mnmax_nyq))
    gmnc = np.zeros((ns, mnmax_nyq))
    bmnc = np.zeros((ns, mnmax_nyq))
    d_bmnc_d_s = np.zeros((ns, mnmax_nyq))
    for jmn, (m, n) in enumerate(zip(xm_nyq, xn_nyq)):
        gmnc[:, jmn] = vs.gmnc[jmn](s)
        bmnc[:, jmn] = vs.bmnc[jmn](s)
        d_bmnc_d_s[:, jmn] = vs.d_bmnc_d_s[jmn](s)
        bsubumnc[:, jmn] = vs.bsubumnc[jmn](s)
        bsubvmnc[:, jmn] = vs.bsubvmnc[jmn](s)
        # Compute the Boozer currents
        if m == 0 and n==0:
            I_Boozer = vs.bsubumnc[jmn](s)
            G_Boozer = vs.bsubvmnc[jmn](s)

    # Note the minus sign. psi in the straight-field-line relation seems to have opposite sign to vmec's phi array.
    edge_toroidal_flux_over_2pi = -vs.phiedge / (2 * np.pi)

    # Save into variables
    variables = ['ns', 's', 'iota', 'd_iota_d_s',
                    'mnmax', 'xm', 'xn', 'rmnc', 'zmns','lmns', 'd_rmnc_d_s', 'd_zmns_d_s', 'd_lmns_d_s',
                    'mnmax_nyq', 'xm_nyq', 'xn_nyq', 'bsubumnc', 'bsubvmnc',
                    'gmnc', 'bmnc', 'd_bmnc_d_s',
                    'I_Boozer', 'G_Boozer',
                    'edge_toroidal_flux_over_2pi']

    #################################
    # EVALUATE ON (theta, phi) GRID #
    #################################
    if full_output:
        # Handle theta
        try:
            ntheta = len(theta)
        except:
            theta = [theta]
        theta_vmec = np.array(theta)
        if theta_vmec.ndim == 1:
            ntheta = len(theta_vmec)
        elif theta_vmec.ndim == 3:
            ntheta = theta_vmec.shape[1]
        else:
            raise ValueError("theta argument must be a float, 1d array, or 3d array.")

        # Handle phi
        try:
            nphi = len(phi)
        except:
            phi = [phi]
        phi = np.array(phi)
        if phi.ndim == 1:
            nphi = len(phi)
        elif phi.ndim == 3:
            nphi = phi.shape[2]
        else:
            raise ValueError("phi argument must be a float, 1d array, or 3d array.")

        # If theta and phi are not already 3D, make them 3D:
        if theta_vmec.ndim == 1:
            theta_vmec = np.kron(np.ones((ns, 1, nphi)), theta_vmec.reshape(1, ntheta, 1))
        if phi.ndim == 1:
            phi = np.kron(np.ones((ns, ntheta, 1)), phi.reshape(1, 1, nphi))

        # Now that we know theta_vmec, compute all the geometric quantities
        angle = xm[:, None, None, None] * theta_vmec[None, :, :, :] - xn[:, None, None, None] * phi[None, :, :, :]
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        # Order of indices in cosangle and sinangle: mn, s, theta, phi
        # Order of indices in rmnc, bmnc, etc: s, mn
        R = np.einsum('ij,jikl->ikl', rmnc, cosangle)
        Z = np.einsum('ij,jikl->ikl', zmns, sinangle)

        # Now handle the Nyquist quantities:
        angle = xm_nyq[:, None, None, None] * theta_vmec[None, :, :, :] - xn_nyq[:, None, None, None] * phi[None, :, :, :]
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        B_sub_theta_vmec = np.einsum('ij,jikl->ikl', bsubumnc, cosangle)
        B_sub_phi = np.einsum('ij,jikl->ikl', bsubvmnc, cosangle)

        # Package results into a structure to return:
        variables += ['R', 'Z', 'B_sub_theta_vmec', 'B_sub_phi']

    #############################
    # SAVE RESULTS IN STRUCTURE #
    #############################
    results = Struct()
    for v in variables:
        results.__setattr__(v, eval(v))

    return results

def clean_0_modes(fmn, xm, xn, n_or_m):
    """
    Eliminate the n or m = 0 modes so that the resulting expressions can then be spectrally inverted.

    Parameters
    ----------
    fmn : array
        The Fourier coefficients of the quantity of interest, ordered as (s, mn).
    xm : array
        The m values corresponding to the mn modes.
    xn : array
        The n values corresponding to the mn modes.
    n_or_m: str
        Whether to eliminate the n=0 modes (if 'n') or the m=0 modes (if 'm').
    """
    #############################
    # SELECT WHICH TO ELIMINATE #
    #############################
    fmn = fmn.copy()
    flag_dim = fmn.ndim > 1
    if n_or_m == 'n':
        ind_0 = np.where(xn == 0)
        if flag_dim:
            fmn[:, ind_0] = 0.0
        else:
            fmn[ind_0] = 0.0
    elif n_or_m == 'm':
        ind_0 = np.where(xm == 0)
        if flag_dim:
            fmn[:, ind_0] = 0.0
        else:
            fmn[ind_0] = 0.0
    else:
        raise ValueError("n_or_m must be 'n' or 'm'.")
    return fmn
        
def select_0_modes(fmn, xm, xn, n_or_m):
    """
    Select only the n or m = 0 modes.

    Parameters
    ----------
    fmn : array
        The Fourier coefficients of the quantity of interest, ordered as (s, mn).
    xm : array
        The m values corresponding to the mn modes.
    xn : array
        The n values corresponding to the mn modes.
    n_or_y : str
        Whether to select the n=0 modes (if 'n') or the m=0 modes (if 'm').
    """
    #############################
    # SELECT WHICH TO ELIMINATE #
    #############################
    fmn = fmn.copy()
    flag_dim = fmn.ndim > 1
    if n_or_m == 'n':
        ind_0 = np.where(xn == 0)
        if flag_dim:
            fmn[:, ind_0] = fmn[:, ind_0]
            fmn[:, np.where(xn != 0)] = 0.0
        else:
            fmn[ind_0] = fmn[ind_0]
            fmn[np.where(xn != 0)] = 0.0
    elif n_or_m == 'm':
        ind_0 = np.where(xm == 0)
        if flag_dim:
            fmn[:, ind_0] = fmn[:, ind_0]
            fmn[:, np.where(xm != 0)] = 0.0
        else:
            fmn[ind_0] = fmn[ind_0]
            fmn[np.where(xm != 0)] = 0.0
    else:
        raise ValueError("n_or_m must be 'n' or 'm'.")
    return fmn

def compute_Phi(results, theta_or_phi = 'phi', test = False):
    """
    Compute the Fourier coefficients of the current potential Phi. This is done by spectrally inverting the equations:
        B_sub_theta = dPhi/dtheta
        B_sub_phi = dPhi/dphi. 
    The resulting Phi is only determined up to a function of theta (if inverting from B_sub_theta) or phi (if inverting from B_sub_phi), 
    which can be fixed by enforcing the other equation.

    Parameters
    ----------
    results : Struct
        The results of vmec_compute_geometry, which contains all the geometric quantities of interest.
    theta_or_phi : str
        Whether to compute Phi by inverting from B_sub_theta (if 'theta') or B_sub_phi (if 'phi').
    test : bool
        If True, then do not enforce the other equation to fix the integration constant. This is useful for testing whether the equations are consistent with each other.
    
    Returns
    -------
    Phimns : array
        The Fourier coefficients of the current potential Phi, ordered as (s, mn).
    """
    #########################
    # SELECT KEY QUANTITIES #
    #########################
    # Magnetic field covariant components
    bsubumnc = results.bsubumnc.copy()
    bsubvmnc = results.bsubvmnc.copy()

    # Check dimensions
    flag_dim = bsubumnc.ndim > 1

    # Modes
    xm = results.xm_nyq
    xn = results.xn_nyq
    
    ####################
    # INVERT THE MODES #
    ####################
    if theta_or_phi == 'theta':
        # Clean the m = 0 modes
        Phimns = clean_0_modes(bsubumnc, xm, xn, 'm')
        # Spectrally invert to get Phi: ignore integration constant, fn of phi
        mask_m = xm != 0
        if flag_dim:
            Phimns[:, mask_m] = bsubumnc[:, mask_m] / xm[mask_m]
        else:
            Phimns[mask_m] = bsubumnc[mask_m] / xm[mask_m]
        if not test:
            DPhins = integrate_spectral(differentiate_spectral(Phimns, xm, xn, 's', 'n') - bsubvmnc, xm, xn, 'c', 'n')
            Phimns = Phimns - DPhins
            
    elif theta_or_phi == 'phi':
        # Clean the n = 0 modes
        Phimns = clean_0_modes(bsubvmnc, xm, xn, 'n')
        # Spectrally invert to get Phi: ignore integration constant, fn of theta
        mask_n = xn != 0
        if flag_dim:
            Phimns[:, mask_n] = bsubvmnc[:, mask_n] / (-xn[mask_n])
        else:
            Phimns[mask_n] = bsubvmnc[mask_n] / (-xn[mask_n])
        if not test:
            DPhims = integrate_spectral(differentiate_spectral(Phimns, xm, xn, 's', 'm') - bsubumnc, xm, xn, 'c', 'm')
            Phimns = Phimns - DPhims
    else:
        raise ValueError("theta_or_phi must be 'theta' or 'phi'.")
    
    return Phimns

class PhiResults(Struct):
    def __init__(self, Phimns, G_Boozer, I_Boozer, xm_nyq, xn_nyq, secular = True):
        # Save the results as attributes of the class
        self.Phimns = Phimns
        self.G_Boozer = G_Boozer
        self.I_Boozer = I_Boozer
        self.xm_nyq = xm_nyq
        self.xn_nyq = xn_nyq
        self.secular = secular

    def evaluate_on_grid(self, theta, phi):
        # Compute periodic part
        Phi_tilde = evaluate_on_grid(self.Phimns, self.xm_nyq, self.xn_nyq, theta, phi, 's')

        if not self.secular:
            return Phi_tilde
        
        # Compute secular part
        Phi_sec = self.I_Boozer * theta[None, :, None] + self.G_Boozer * phi[None, None, :]

        return Phi_tilde + Phi_sec

def integrate_spectral(fmn, xm, xn, s_or_c = 's', m_or_n = 'm'):
    """
    Integrate a quantity given by Fourier coefficients fmn in spectral space. Ignores the constant of integration.

    Parameters
    ----------
    fmn : array
        The Fourier coefficients of the quantity of interest, ordered as (s, mn).
    xm : array
        The m values corresponding to the mn modes.
    xn : array
        The n values corresponding to the mn modes.
    s_or_c : str
        Whether the Fourier coefficients are in terms of sine or cosine modes. If 's', then the modes are sine modes and the angle is xm*theta - xn*phi. If 'c', then the modes are cosine modes and the angle is xm*theta - xn*phi.
    m_or_n : str
        Whether to integrate with respect to theta (if 'm') or phi (if 'n'). 

    Returns
    -------
    fmn_integrated : array
        The integrated quantity in spectral space, ordered as (s, mn).
    """
    fmn = np.asarray(fmn)
    if s_or_c == 's':
        if m_or_n == 'm':
            denom = -xm
        elif m_or_n == 'n':
            denom = xn
        else:
            raise ValueError("m_or_n must be 'm' or 'n'.")
    elif s_or_c == 'c':
        if m_or_n == 'm':
            denom = xm
        elif m_or_n == 'n':
            denom = -xn
        else:
            raise ValueError("m_or_n must be 'm' or 'n'.")
    else:
        raise ValueError("s_or_c must be 's' or 'c'.")

    inv_denom = np.zeros_like(denom, dtype=fmn.dtype)
    mask = denom != 0
    inv_denom[mask] = 1.0 / denom[mask]

    if fmn.ndim == 2:
        return fmn * inv_denom[None, :]
    if fmn.ndim == 1:
        return fmn * inv_denom
    raise ValueError("fmn must be either 1D or 2D array.")

def differentiate_spectral(fmn, xm, xn, s_or_c = 's', m_or_n = 'm'):
    """
    Differentiate a quantity given by Fourier coefficients fmn in spectral space.

    Parameters
    ----------
    fmn : array
        The Fourier coefficients of the quantity of interest, ordered as (s, mn) or (mn,).
    xm : array
        The m values corresponding to the mn modes.
    xn : array
        The n values corresponding to the mn modes.
    s_or_c : str
        Whether the Fourier coefficients are in terms of sine or cosine modes. If 's', then the modes are sine modes and the angle is xm*theta - xn*phi. If 'c', then the modes are cosine modes and the angle is xm*theta - xn*phi.
    m_or_n : str
        Whether to differentiate with respect to theta (if 'm') or phi (if 'n').
        
    Returns
    -------
    fmn_differentiated : array
        The differentiated quantity in spectral space, ordered as (s, mn).
    """
    fmn = np.asarray(fmn)
    if s_or_c == 's':
        if m_or_n == 'm':
            coeff = xm
        elif m_or_n == 'n':
            coeff = -xn
        else:
            raise ValueError("m_or_n must be 'm' or 'n'.")
    elif s_or_c == 'c':
        if m_or_n == 'm':
            coeff = -xm
        elif m_or_n == 'n':
            coeff = xn
        else:
            raise ValueError("m_or_n must be 'm' or 'n'.")
    else:
        raise ValueError("s_or_c must be 's' or 'c'.")

    if fmn.ndim == 2:
        return fmn * coeff[None, :]
    if fmn.ndim == 1:
        return fmn * coeff
    raise ValueError("fmn must be either 1D or 2D array.")

def evaluate_on_grid(fmn, xm, xn, theta, phi, s_or_c = 's'):
    """
    Evaluate a quantity given by Fourier coefficients fmn on a grid of theta and phi.

    Parameters
    ----------
    fmn : array
        The Fourier coefficients of the quantity of interest, ordered as (s, mn) or (mn).
    xm : array
        The m values corresponding to the mn modes.
    xn : array
        The n values corresponding to the mn modes.
    theta : array
        The theta values on which to evaluate the quantity.
    phi : array
        The phi values on which to evaluate the quantity.
    s_or_c : str
        Whether the Fourier coefficients are in terms of sine or cosine modes. If 's', then the modes are sine modes and the angle is xm*theta - xn*phi. If 'c', then the modes are cosine modes and the angle is xm*theta - xn*phi.

    Returns
    -------
    f_grid : array
        The quantity evaluated on the (theta, phi) grid.
    """
    # Compute the angle
    angle = xm[:, None, None] * theta[None, :, None] - xn[:, None, None] * phi[None, None, :]

    # Check if s dimension is present in fmn
    if fmn.ndim == 2:
        # fmn has shape (s, mn)
        if s_or_c == 's':
            f_grid = np.einsum('ij,jkl->ikl', fmn, np.sin(angle))
        elif s_or_c == 'c': 
            f_grid = np.einsum('ij,jkl->ikl', fmn, np.cos(angle))
        else:
            raise ValueError("s_or_c must be 's' or 'c'.")
    elif fmn.ndim == 1:
        # fmn has shape (mn,)
        if s_or_c == 's':
            f_grid = np.einsum('j,jkl->kl', fmn, np.sin(angle))
        elif s_or_c == 'c': 
            f_grid = np.einsum('j,jkl->kl', fmn, np.cos(angle))
        else:
            raise ValueError("s_or_c must be 's' or 'c'.")
    else:
        raise ValueError("fmn must be either 1D or 2D array.")
    return f_grid

def load_vmec(vmec_file=None):
    if vmec_file is None:
        vmec_file = os.environ.get("VMEC_WOUT_FILE")
    if vmec_file is None:
        raise ValueError("vmec_file must be provided or VMEC_WOUT_FILE must be set.")
    vmec = Vmec(vmec_file)
    vmec.run()
    vs = vmec_splines(vmec)
    return vs

def evaluate_contour_RZ(theta_contour, phi_contour, results, full_output = True):
    """
    Evaluate R, Z and their first three total derivatives with respect to theta
    along a contour (theta(t), phi(t)) using spectral coefficients.

    The total derivative uses the chain rule:
        dR/dtheta = dR/dtheta_partial + dR/dphi_partial * dphi/dtheta
    and similarly for higher orders.

    Parameters
    ----------
    theta_contour : array, shape (n_points,)
        The theta values along the contour.
    phi_contour : array, shape (n_points,)
        The phi values along the contour.
    results : Struct
        The results of vmec_compute_geometry (must contain rmnc, zmns, xm, xn,
        bsubumnc, bsubvmnc, xm_nyq, xn_nyq).
    full_output : bool
        If True, then also return the partial derivatives and the dphi/dtheta and its derivatives used in the chain rule. If False, then only return R, Z and their total derivatives.

    Returns
    -------
    R, Z : arrays, shape (n_points,)
        Position along the contour.
    dR, dZ, dphi : arrays, shape (n_points,) (optional)
        First total derivatives with respect to theta.
    d2R, d2Z, d2phi : arrays, shape (n_points,) (optional)
        Second total derivatives with respect to theta.
    d3R, d3Z, d3phi : arrays, shape (n_points,) (optional)
        Third total derivatives with respect to theta.
    """
    # Check that all necessary quantities are present in results
    required_quantities = ['rmnc', 'zmns', 'xm', 'xn', 'bsubumnc', 'bsubvmnc', 'xm_nyq', 'xn_nyq']
    for q in required_quantities:
        if not hasattr(results, q):
            raise ValueError(f"results must contain {q} to evaluate contour R, Z.")

    # Obtain the shape 
    rmnc = results.rmnc[0, :] if np.shape(results.rmnc) == 2 else results.rmnc   # (mnmax,)
    zmns = results.zmns[0, :] if np.shape(results.zmns) == 2 else results.zmns   # (mnmax,)
    xm = results.xm
    xn = results.xn

    # Check same shape theta_contour and phi_contour
    if theta_contour.shape != phi_contour.shape:
        raise ValueError("theta_contour and phi_contour must have the same shape.")
    
    shape_grid = theta_contour.shape
    theta_contour = theta_contour.flatten()
    phi_contour = phi_contour.flatten()

    # Angle for each mode at each contour point: shape (mnmax, n_points)
    angle = xm[:, None] * theta_contour[None, :] - xn[:, None] * phi_contour[None, :]
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)

    # --- R, Z on contour ---
    R = np.dot(rmnc, cosangle)     # (n_points,)
    Z = np.dot(zmns, sinangle)

    if not full_output:
        return R, Z
    
    # --- Partial derivatives of R, Z wrt theta and phi ---
    #   R = sum rmnc * cos(m*theta - n*phi)
    #   dR/dtheta_partial = sum rmnc * (-m) * sin(m*theta - n*phi)
    #   dR/dphi_partial   = sum rmnc * ( n) * sin(m*theta - n*phi)
    dR_dt = np.dot(rmnc * (-xm), sinangle)
    dR_dp = np.dot(rmnc * ( xn), sinangle)
    dZ_dt = np.dot(zmns * ( xm), cosangle)
    dZ_dp = np.dot(zmns * (-xn), cosangle)

    # --- Second partial derivatives ---
    d2R_dt2  = np.dot(rmnc * (-xm**2), cosangle)
    d2R_dtdp = np.dot(rmnc * ( xm*xn), cosangle)
    d2R_dp2  = np.dot(rmnc * (-xn**2), cosangle)
    d2Z_dt2  = np.dot(zmns * (-xm**2), sinangle)
    d2Z_dtdp = np.dot(zmns * ( xm*xn), sinangle)
    d2Z_dp2  = np.dot(zmns * (-xn**2), sinangle)

    # --- Third partial derivatives ---
    d3R_dt3   = np.dot(rmnc * ( xm**3), sinangle)
    d3R_dt2dp = np.dot(rmnc * (-xm**2 * xn), sinangle)
    d3R_dtdp2 = np.dot(rmnc * ( xm * xn**2), sinangle)
    d3R_dp3   = np.dot(rmnc * (-xn**3), sinangle)
    d3Z_dt3   = np.dot(zmns * ( xm**3), cosangle)
    d3Z_dt2dp = np.dot(zmns * (-xm**2 * xn), cosangle)
    d3Z_dtdp2 = np.dot(zmns * ( xm * xn**2), cosangle)
    d3Z_dp3   = np.dot(zmns * (-xn**3), cosangle)

    # --- dphi/dtheta from the ODE (spectral) ---
    bsubu = results.bsubumnc[0, :] if np.shape(results.bsubumnc) == 2 else results.bsubumnc
    bsubv = results.bsubvmnc[0, :] if np.shape(results.bsubvmnc) == 2 else results.bsubvmnc
    xm_nyq = results.xm_nyq
    xn_nyq = results.xn_nyq
    angle_nyq = xm_nyq[:, None] * theta_contour[None, :] - xn_nyq[:, None] * phi_contour[None, :]
    cosangle_nyq = np.cos(angle_nyq)
    B_sub_theta = np.dot(bsubu, cosangle_nyq)  # (n_points,)
    B_sub_phi   = np.dot(bsubv, cosangle_nyq)
    dphi = -B_sub_theta / B_sub_phi            # dphi/dtheta

    # --- Total first derivatives (chain rule) ---
    dR = dR_dt + dR_dp * dphi
    dZ = dZ_dt + dZ_dp * dphi

    # --- d2phi/dtheta2 from differentiating dphi/dtheta = -Bt/Bp ---
    #   d(Bt)/dtheta_total = dBt/dt + dBt/dp * dphi
    sinangle_nyq = np.sin(angle_nyq)
    dBt_dt = np.dot(bsubu * (-xm_nyq), sinangle_nyq)
    dBt_dp = np.dot(bsubu * ( xn_nyq), sinangle_nyq)
    dBp_dt = np.dot(bsubv * (-xm_nyq), sinangle_nyq)
    dBp_dp = np.dot(bsubv * ( xn_nyq), sinangle_nyq)
    dBt = dBt_dt + dBt_dp * dphi
    dBp = dBp_dt + dBp_dp * dphi
    d2phi = -(dBt * B_sub_phi - B_sub_theta * dBp) / B_sub_phi**2

    # --- Total second derivatives ---
    d2R = d2R_dt2 + 2 * d2R_dtdp * dphi + d2R_dp2 * dphi**2 + dR_dp * d2phi
    d2Z = d2Z_dt2 + 2 * d2Z_dtdp * dphi + d2Z_dp2 * dphi**2 + dZ_dp * d2phi

    # --- d3phi/dtheta3 ---
    d2Bt_dt2  = np.dot(bsubu * (-xm_nyq**2), cosangle_nyq)
    d2Bt_dtdp = np.dot(bsubu * ( xm_nyq*xn_nyq), cosangle_nyq)
    d2Bt_dp2  = np.dot(bsubu * (-xn_nyq**2), cosangle_nyq)
    d2Bp_dt2  = np.dot(bsubv * (-xm_nyq**2), cosangle_nyq)
    d2Bp_dtdp = np.dot(bsubv * ( xm_nyq*xn_nyq), cosangle_nyq)
    d2Bp_dp2  = np.dot(bsubv * (-xn_nyq**2), cosangle_nyq)
    d2Bt = d2Bt_dt2 + 2 * d2Bt_dtdp * dphi + d2Bt_dp2 * dphi**2 + dBt_dp * d2phi
    d2Bp = d2Bp_dt2 + 2 * d2Bp_dtdp * dphi + d2Bp_dp2 * dphi**2 + dBp_dp * d2phi
    d3phi = -(d2Bt * B_sub_phi - B_sub_theta * d2Bp
              - 2 * dBp * (dBt * B_sub_phi - B_sub_theta * dBp) / B_sub_phi) / B_sub_phi**2

    # --- Total third derivatives ---
    d3R = (d3R_dt3 + 3 * d3R_dt2dp * dphi + 3 * d3R_dtdp2 * dphi**2 + d3R_dp3 * dphi**3
           + 3 * (d2R_dtdp + d2R_dp2 * dphi) * d2phi + dR_dp * d3phi)
    d3Z = (d3Z_dt3 + 3 * d3Z_dt2dp * dphi + 3 * d3Z_dtdp2 * dphi**2 + d3Z_dp3 * dphi**3
           + 3 * (d2Z_dtdp + d2Z_dp2 * dphi) * d2phi + dZ_dp * d3phi)
    
    # Reshape outputs to original contour shape
    R = R.reshape(shape_grid)
    Z = Z.reshape(shape_grid)
    dR = dR.reshape(shape_grid)
    dZ = dZ.reshape(shape_grid)
    dphi = dphi.reshape(shape_grid)
    d2R = d2R.reshape(shape_grid)
    d2Z = d2Z.reshape(shape_grid)
    d2phi = d2phi.reshape(shape_grid)
    d3R = d3R.reshape(shape_grid)
    d3Z = d3Z.reshape(shape_grid)
    d3phi = d3phi.reshape(shape_grid)

    return R, Z, dR, dZ, dphi, d2R, d2Z, d2phi, d3R, d3Z, d3phi

def evaluate_surface_geometry(theta_grid, phi_grid, results):
    """
    Evaluate R, Z and typical surface geometry such as the normal vector, the surface mean and Gaussian curvature.

    Parameters
    ----------
    theta_grid : array
        The theta values on the grid.
    phi_grid : array
        The phi values on the grid.
    results : Struct
        The results of vmec_compute_geometry (must contain rmnc, zmns, xm, xn, grad_psi, xm_nyq, xn_nyq).

    Returns
    -------
    R : array
        The R values on the grid.
    Z : array
        The Z values on the grid.
    """
    # Check that all necessary quantities are present in results
    required_quantities = ['rmnc', 'zmns', 'xm', 'xn']
    for q in required_quantities:
        if not hasattr(results, q):
            raise ValueError(f"results must contain {q} to evaluate surface geometry.")

    # Check the grids same shape 
    assert theta_grid.shape == phi_grid.shape, "theta_grid and phi_grid must have the same shape."
    grid_shape = theta_grid.shape

    # Flatten the grids
    theta_grid = theta_grid.flatten()
    phi_grid = phi_grid.flatten()

    # Extract data
    rmnc = results.rmnc[0, :] if results.rmnc.ndim == 2 else results.rmnc  # (mnmax,)
    zmns = results.zmns[0, :] if results.zmns.ndim == 2 else results.zmns  # (mnmax,)
    xm = results.xm
    xn = results.xn

    # Angle for each mode at each grid point: shape (mnmax, **) where ** is the shape of the theta_grid and phi_grid
    angle = xm[:, None] * theta_grid[None, :] - xn[:, None] * phi_grid[None, :]
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)

    # R and Z on grid
    R = np.dot(rmnc, cosangle)     # (n_theta, n_phi)
    Z = np.dot(zmns, sinangle)

    # --- Spectral derivatives ---
    # dR/dtheta, dR/dphi, dZ/dtheta, dZ/dphi
    dR_dtheta = np.dot(rmnc * xm, -sinangle)  # (n_theta, n_phi)
    dR_dphi   = np.dot(rmnc * (-xn), -sinangle)
    dZ_dtheta = np.dot(zmns * xm,  cosangle)
    dZ_dphi   = np.dot(zmns * (-xn),  cosangle)

    # dX/dtheta, dX/dphi, dY/dtheta, dY/dphi
    dX_dtheta = dR_dtheta * np.cos(phi_grid)
    dX_dphi   = dR_dphi * np.cos(phi_grid) - R * np.sin(phi_grid)
    dY_dtheta = dR_dtheta * np.sin(phi_grid)
    dY_dphi   = dR_dphi * np.sin(phi_grid) + R * np.cos(phi_grid)
    dZ_dtheta = dZ_dtheta
    dZ_dphi   = dZ_dphi

    # Tangent vectors
    e_theta = np.stack([dX_dtheta, dY_dtheta, dZ_dtheta], axis=-1)
    e_phi   = np.stack([dX_dphi,   dY_dphi,   dZ_dphi],   axis=-1)

    # Surface normal (not normalized)
    normal = np.cross(e_theta, e_phi)
    norm_normal = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_unit = normal / norm_normal

    # First fundamental form (metric tensor):
    # I = [[E, F], [F, G]]
    # where E = e_theta . e_theta, F = e_theta . e_phi, G = e_phi . e_phi
    # (i.e., the metric coefficients for the surface)
    E = np.sum(e_theta * e_theta, axis=-1)
    F = np.sum(e_theta * e_phi, axis=-1)
    G = np.sum(e_phi * e_phi, axis=-1)
    sqrt_g = np.sqrt(E * G - F**2)

    # --- Second derivatives for curvatures ---
    # d2R/dtheta2, d2R/dphidtheta, d2R/dphi2, etc.
    d2R_dtheta2 = np.dot(rmnc * xm**2, -cosangle)
    d2R_dphidtheta = np.dot(rmnc * xm * (-xn), -cosangle)
    d2R_dphi2 = np.dot(rmnc * xn**2, -cosangle)
    d2Z_dtheta2 = np.dot(zmns * xm**2, -sinangle)
    d2Z_dphidtheta = np.dot(zmns * xm * (-xn), -sinangle)
    d2Z_dphi2 = np.dot(zmns * xn**2, -sinangle)

    # d2X, d2Y, d2Z
    d2X_dtheta2 = d2R_dtheta2 * np.cos(phi_grid)
    d2X_dphidtheta = d2R_dphidtheta * np.cos(phi_grid) - dR_dtheta * np.sin(phi_grid)
    d2X_dphi2 = d2R_dphi2 * np.cos(phi_grid) - 2 * dR_dphi * np.sin(phi_grid) - R * np.cos(phi_grid)

    d2Y_dtheta2 = d2R_dtheta2 * np.sin(phi_grid)
    d2Y_dphidtheta = d2R_dphidtheta * np.sin(phi_grid) + dR_dtheta * np.cos(phi_grid)
    d2Y_dphi2 = d2R_dphi2 * np.sin(phi_grid) + 2 * dR_dphi * np.cos(phi_grid) - R * np.sin(phi_grid)

    d2Z_dtheta2 = d2Z_dtheta2
    d2Z_dphidtheta = d2Z_dphidtheta
    d2Z_dphi2 = d2Z_dphi2

    # Second derivatives in 3D
    r_theta_theta = np.stack([d2X_dtheta2, d2Y_dtheta2, d2Z_dtheta2], axis=-1)
    r_theta_phi   = np.stack([d2X_dphidtheta, d2Y_dphidtheta, d2Z_dphidtheta], axis=-1)
    r_phi_phi     = np.stack([d2X_dphi2, d2Y_dphi2, d2Z_dphi2], axis=-1)

    # Second fundamental form coefficients: L, M, N
    # II = [[L, M], [M, N]] 
    # where L = r_theta_theta . n, M = r_theta_phi . n, N = r_phi_phi . n
    n = normal_unit
    L = np.sum(r_theta_theta * n, axis=-1)
    M = np.sum(r_theta_phi * n, axis=-1)
    N = np.sum(r_phi_phi * n, axis=-1)

    # Gaussian curvature: det II / det I = (L*N - M^2) / (E*G - F^2)
    gaussian_curvature = (L*N - M**2) / (E*G - F**2)

    # Mean curvature: H = 0.5 * trace(II * I^-1) = 0.5 * ( (E*N - 2*F*M + G*L) / (E*G - F^2) )
    mean_curvature = 0.5 * ( (E*N - 2*F*M + G*L) / (E*G - F**2) )

    # Reshape outputs to grid shape
    R = R.reshape(grid_shape)
    Z = Z.reshape(grid_shape)
    normal_unit = normal_unit.reshape(grid_shape + (3,))
    gaussian_curvature = gaussian_curvature.reshape(grid_shape)
    mean_curvature = mean_curvature.reshape(grid_shape)

    return R, Z, normal_unit, gaussian_curvature, mean_curvature

def evaluate_curvature_B_field(theta_grid, phi_grid, results):
    """
    Evaluate the curvature of the magnetic field lines on the surface.

    Parameters
    ----------
    theta_grid : array
        The theta values on the grid.
    phi_grid : array
        The phi values on the grid.
    vs : Vmec or vmec_splines
        The Vmec object or vmec_splines containing the necessary data.
    s : array
        The radial coordinate(s) at which to evaluate the curvature.

    Returns
    -------
    curvature_B : array
        The curvature of the magnetic field lines on the surface.
    normalized_curvature_B : array
        The curvature of the magnetic field lines normalized by the magnetic field strength.
    """
    # Shorthand:
    ns = len(results.s)
    mnmax = results.mnmax
    xm = results.xm
    xn = results.xn
    mnmax_nyq = results.mnmax_nyq
    xm_nyq = results.xm_nyq
    xn_nyq = results.xn_nyq
    iota = results.iota
    d_iota_d_s = results.d_iota_d_s
    rmnc = results.rmnc
    zmns = results.zmns
    lmns = results.lmns
    d_rmnc_d_s = results.d_rmnc_d_s
    d_zmns_d_s = results.d_zmns_d_s
    d_lmns_d_s = results.d_lmns_d_s
    gmnc = results.gmnc
    bmnc = results.bmnc
    d_bmnc_d_s = results.d_bmnc_d_s
    edge_toroidal_flux_over_2pi = results.edge_toroidal_flux_over_2pi

    # Check the grids same shape
    assert theta_grid.shape == phi_grid.shape, "theta_grid and phi_grid must have the same shape."
    grid_shape = theta_grid.shape

    # Flatten the grids
    theta_grid = theta_grid.flatten()
    phi_grid = phi_grid.flatten()

    # Now that we know theta_vmec, compute all the geometric quantities
    angle = xm[:, None] * theta_grid[None, :] - xn[:, None] * phi_grid[None, :]
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm[:, None] * cosangle
    ncosangle = xn[:, None] * cosangle
    msinangle = xm[:, None] * sinangle
    nsinangle = xn[:, None] * sinangle
    # Order of indices in cosangle and sinangle: mn, s, theta, phi
    # Order of indices in rmnc, bmnc, etc: s, mn
    R = np.einsum('ij,jk->ik', rmnc, cosangle)
    d_R_d_s = np.einsum('ij,jk->ik', d_rmnc_d_s, cosangle)
    d_R_d_theta_vmec = -np.einsum('ij,jk->ik', rmnc, msinangle)
    d_R_d_phi = np.einsum('ij,jk->ik', rmnc, nsinangle)

    Z = np.einsum('ij,jk->ik', zmns, sinangle)
    d_Z_d_s = np.einsum('ij,jk->ik', d_zmns_d_s, sinangle)
    d_Z_d_theta_vmec = np.einsum('ij,jk->ik', zmns, mcosangle)
    d_Z_d_phi = -np.einsum('ij,jk->ik', zmns, ncosangle)

    d_lambda_d_theta_vmec = np.einsum('ij,jk->ik', lmns, mcosangle)
    d_lambda_d_phi = -np.einsum('ij,jk->ik', lmns, ncosangle)

    # Now handle the Nyquist quantities:
    angle = xm_nyq[:, None] * theta_grid[None, :] - xn_nyq[:, None] * phi_grid[None, :]
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm_nyq[:, None] * cosangle
    ncosangle = xn_nyq[:, None] * cosangle
    msinangle = xm_nyq[:, None] * sinangle
    nsinangle = xn_nyq[:, None] * sinangle

    sqrt_g_vmec = np.einsum('ij,jk->ik', gmnc, cosangle)
    B_magnitude = np.einsum('ij,jk->ik', bmnc, cosangle)
    d_B_d_s = np.einsum('ij,jk->ik', d_bmnc_d_s, cosangle)
    d_B_d_theta_vmec = -np.einsum('ij,jk->ik', bmnc, msinangle)
    d_B_d_phi = np.einsum('ij,jk->ik', bmnc, nsinangle)

    # *********************************************************************
    # Using R(theta,phi) and Z(theta,phi), compute the Cartesian
    # components of the gradient basis vectors using the dual relations:
    # *********************************************************************
    sinphi = np.sin(phi_grid)
    cosphi = np.cos(phi_grid)
    X = R * cosphi
    d_X_d_theta_vmec = d_R_d_theta_vmec * cosphi
    d_X_d_phi = d_R_d_phi * cosphi - R * sinphi
    d_X_d_s = d_R_d_s * cosphi
    Y = R * sinphi
    d_Y_d_theta_vmec = d_R_d_theta_vmec * sinphi
    d_Y_d_phi = d_R_d_phi * sinphi + R * cosphi
    d_Y_d_s = d_R_d_s * sinphi

    # Now use the dual relations to get the Cartesian components of grad s, grad theta_vmec, and grad phi:
    grad_s_X = (d_Y_d_theta_vmec * d_Z_d_phi - d_Z_d_theta_vmec * d_Y_d_phi) / sqrt_g_vmec
    grad_s_Y = (d_Z_d_theta_vmec * d_X_d_phi - d_X_d_theta_vmec * d_Z_d_phi) / sqrt_g_vmec
    grad_s_Z = (d_X_d_theta_vmec * d_Y_d_phi - d_Y_d_theta_vmec * d_X_d_phi) / sqrt_g_vmec

    grad_theta_vmec_X = (d_Y_d_phi * d_Z_d_s - d_Z_d_phi * d_Y_d_s) / sqrt_g_vmec
    grad_theta_vmec_Y = (d_Z_d_phi * d_X_d_s - d_X_d_phi * d_Z_d_s) / sqrt_g_vmec
    grad_theta_vmec_Z = (d_X_d_phi * d_Y_d_s - d_Y_d_phi * d_X_d_s) / sqrt_g_vmec

    grad_phi_X = (d_Y_d_s * d_Z_d_theta_vmec - d_Z_d_s * d_Y_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Y = (d_Z_d_s * d_X_d_theta_vmec - d_X_d_s * d_Z_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Z = (d_X_d_s * d_Y_d_theta_vmec - d_Y_d_s * d_X_d_theta_vmec) / sqrt_g_vmec
    # End of dual relations.

    # *********************************************************************
    # Compute the Cartesian components of other quantities we need:
    # *********************************************************************
    grad_B_X = d_B_d_s * grad_s_X + d_B_d_theta_vmec * grad_theta_vmec_X + d_B_d_phi * grad_phi_X
    grad_B_Y = d_B_d_s * grad_s_Y + d_B_d_theta_vmec * grad_theta_vmec_Y + d_B_d_phi * grad_phi_Y
    grad_B_Z = d_B_d_s * grad_s_Z + d_B_d_theta_vmec * grad_theta_vmec_Z + d_B_d_phi * grad_phi_Z

    B_X = edge_toroidal_flux_over_2pi * ((1 + d_lambda_d_theta_vmec) * d_X_d_phi + (iota[:, None, None] - d_lambda_d_phi) * d_X_d_theta_vmec) / sqrt_g_vmec
    B_Y = edge_toroidal_flux_over_2pi * ((1 + d_lambda_d_theta_vmec) * d_Y_d_phi + (iota[:, None, None] - d_lambda_d_phi) * d_Y_d_theta_vmec) / sqrt_g_vmec
    B_Z = edge_toroidal_flux_over_2pi * ((1 + d_lambda_d_theta_vmec) * d_Z_d_phi + (iota[:, None, None] - d_lambda_d_phi) * d_Z_d_theta_vmec) / sqrt_g_vmec


    # ************************************************
    # Compute fieldline curvature: kappa = b . grad b = (grad(|B|^2/2) - b (b . grad (|B|^2/2))) / B^2 (for vacuum)
    # where b = B / |B| is the unit vector along the magnetic field.
    # ************************************************
    b_X = B_X / B_magnitude
    b_Y = B_Y / B_magnitude
    b_Z = B_Z / B_magnitude 
    b_dot_grad_B = b_X * grad_B_X + b_Y * grad_B_Y + b_Z * grad_B_Z
    curvature_B_X = (grad_B_X - b_X * b_dot_grad_B) / B_magnitude**2
    curvature_B_Y = (grad_B_Y - b_Y * b_dot_grad_B) / B_magnitude**2
    curvature_B_Z = (grad_B_Z - b_Z * b_dot_grad_B) / B_magnitude**2
    curvature_B = np.sqrt(curvature_B_X**2 + curvature_B_Y**2 + curvature_B_Z**2)

    normal_unit = np.stack([curvature_B_X, curvature_B_Y, curvature_B_Z], axis=-1)/curvature_B[..., None]

    # Reshape outputs to grid shape
    curvature_B = curvature_B.reshape(grid_shape)
    normal_unit = normal_unit.reshape(grid_shape + (3,))

    return curvature_B, normal_unit

def plot_3D_curve(R, Z, phi, ax=None):
    """
    Plot a curve in 3D space given by R, Z and phi.

    Parameters
    ----------
    R : array
        The R values of the curve, ordered as (n_points,).
    Z : array
        The Z values of the curve, ordered as (n_points,).
    phi : array
        The phi values of the curve, ordered as (n_points,).
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = Z
    ax.plot(x, y, z, 'k-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def characterise_curve(R, Z, phi, theta, dR=None, dZ=None, dphi_dtheta=None,
                       d2R=None, d2Z=None, d2phi_dtheta=None,
                       d3R=None, d3Z=None, d3phi_dtheta=None,
                       surface_normal = None):
    """
    Given a curve in R-Z-phi space parametrised by theta, compute the curvature and torsion of the curve.

    If spectral derivatives (dR, dZ, dphi_dtheta, ...) are provided they are
    used directly; otherwise derivatives are estimated with np.gradient.

    Parameters
    ----------
    R : array
        The R values of the curve, ordered as (n_points,).
    Z : array
        The Z values of the curve, ordered as (n_points,).
    phi : array
        The phi values of the curve, ordered as (n_points,).
    theta : array
        The theta values of the curve, ordered as (n_points,).
    dR, dZ, dphi_dtheta : array, optional
        First derivatives with respect to theta.
    d2R, d2Z, d2phi_dtheta : array, optional
        Second derivatives with respect to theta.
    d3R, d3Z, d3phi_dtheta : array, optional
        Third derivatives with respect to theta.
    Returns
    -------
    curvature : array
        The curvature of the curve, ordered as (n_points,).
    torsion : array
        The torsion of the curve, ordered as (n_points,).
    """
    # Compute derivatives with respect to theta
    if dR is None:
        dR_dtheta = np.gradient(R, theta)
        dZ_dtheta = np.gradient(Z, theta)
        dphi_dt   = np.gradient(phi, theta)
    else:
        dR_dtheta = dR
        dZ_dtheta = dZ
        dphi_dt   = dphi_dtheta

    if d2R is None:
        d2R_dtheta2   = np.gradient(dR_dtheta, theta)
        d2Z_dtheta2   = np.gradient(dZ_dtheta, theta)
        d2phi_dtheta2 = np.gradient(dphi_dt, theta)
    else:
        d2R_dtheta2   = d2R
        d2Z_dtheta2   = d2Z
        d2phi_dtheta2 = d2phi_dtheta

    if d3R is None:
        d3R_dtheta3   = np.gradient(d2R_dtheta2, theta)
        d3Z_dtheta3   = np.gradient(d2Z_dtheta2, theta)
        d3phi_dtheta3 = np.gradient(d2phi_dtheta2, theta)
    else:
        d3R_dtheta3   = d3R
        d3Z_dtheta3   = d3Z
        d3phi_dtheta3 = d3phi_dtheta

    # Shape check
    assert phi.shape == theta.shape, 'phi and theta must have the same shape.'

    # Flatten arrays for easier calculations
    grid_shape = theta.shape
    R = R.flatten()
    Z = Z.flatten()
    phi = phi.flatten()
    dR_dtheta = dR_dtheta.flatten()
    dZ_dtheta = dZ_dtheta.flatten()
    dphi_dt = dphi_dt.flatten()
    d2R_dtheta2 = d2R_dtheta2.flatten()
    d2Z_dtheta2 = d2Z_dtheta2.flatten()
    d2phi_dtheta2 = d2phi_dtheta2.flatten()
    d3R_dtheta3 = d3R_dtheta3.flatten()
    d3Z_dtheta3 = d3Z_dtheta3.flatten()
    d3phi_dtheta3 = d3phi_dtheta3.flatten()


    # Compute curvature and torsion using the formulas:
    # curvature = |r' x r''| / |r'|^3
    # torsion = (r' x r'') . r''' / |r' x r''|^2

    # Compute r' in 3D space (X, Y, Z) where X = R*cos(phi), Y = R*sin(phi), Z = Z
    dX_dtheta = dR_dtheta * np.cos(phi) - R * np.sin(phi) * dphi_dt
    dY_dtheta = dR_dtheta * np.sin(phi) + R * np.cos(phi) * dphi_dt
    r_prime = np.stack((dX_dtheta, dY_dtheta, dZ_dtheta), axis=-1)

    # Compute r'' in 3D space using the product rule
    d2X_dtheta2 = d2R_dtheta2 * np.cos(phi) - 2 * dR_dtheta * np.sin(phi) * dphi_dt - R * np.cos(phi) * dphi_dt**2 - R * np.sin(phi) * d2phi_dtheta2
    d2Y_dtheta2 = d2R_dtheta2 * np.sin(phi) + 2 * dR_dtheta * np.cos(phi) * dphi_dt - R * np.sin(phi) * dphi_dt**2 + R * np.cos(phi) * d2phi_dtheta2
    r_double_prime = np.stack((d2X_dtheta2, d2Y_dtheta2, d2Z_dtheta2), axis=-1)

    # Compute r''' in 3D space using the product rule again
    d3X_dtheta3 = d3R_dtheta3 * np.cos(phi) - 3 * d2R_dtheta2 * np.sin(phi) * dphi_dt - 3 * dR_dtheta * np.cos(phi) * dphi_dt**2 + R * np.sin(phi) * dphi_dt**3 - 3 * dR_dtheta * np.sin(phi) * d2phi_dtheta2 - 3 * R * np.cos(phi) * dphi_dt * d2phi_dtheta2 - R * np.sin(phi) * d3phi_dtheta3
    d3Y_dtheta3 = d3R_dtheta3 * np.sin(phi) + 3 * d2R_dtheta2 * np.cos(phi) * dphi_dt - 3 * dR_dtheta * np.sin(phi) * dphi_dt**2 - R * np.cos(phi) * dphi_dt**3 + 3 * dR_dtheta * np.cos(phi) * d2phi_dtheta2 - 3 * R * np.sin(phi) * dphi_dt * d2phi_dtheta2 + R * np.cos(phi) * d3phi_dtheta3
    r_triple_prime = np.stack((d3X_dtheta3, d3Y_dtheta3, d3Z_dtheta3), axis=-1)

    # Tangent vector is r' normalized
    d_ell_d_theta = np.linalg.norm(r_prime, axis=-1)
    tangent_unit = r_prime / d_ell_d_theta[:, None]

    # Compute the binormal vector v = r' x r''
    cross_product = np.cross(r_prime, r_double_prime)
    norm_cross = np.linalg.norm(cross_product, axis=-1)
    binormal_unit = cross_product / norm_cross[:, None]

    # Compute the curvature
    curvature = norm_cross / d_ell_d_theta**3
    normal_unit = np.cross(binormal_unit, tangent_unit)

    # Compute the torsion
    torsion = np.einsum('ij,ij->i', cross_product, r_triple_prime) / norm_cross**2

    # Surface properties
    if surface_normal is not None:
        # Flatten first dim of surface_normal
        surface_normal = surface_normal.reshape(-1, 3)

        # Normal curvature: k_n = curvature * (normal_unit . surface_normal)
        normal_curvature = curvature * np.einsum('ij,ij->i', normal_unit, surface_normal)

        # Geodesic curvature: k_g = sqrt(curvature^2 - k_n^2)
        geodesic_curvature = np.sqrt(curvature**2 - normal_curvature**2)

        # Reshape outputs to original grid shape
        curvature = curvature.reshape(grid_shape)
        torsion = torsion.reshape(grid_shape)
        normal_curvature = normal_curvature.reshape(grid_shape)
        geodesic_curvature = geodesic_curvature.reshape(grid_shape)

        return curvature, torsion, normal_curvature, geodesic_curvature
    
    # Reshape outputs to original grid shape
    curvature = curvature.reshape(grid_shape)
    torsion = torsion.reshape(grid_shape)

    return curvature, torsion

def _dphi_dtheta_rhs(theta, phi, xm, xn, bsubu, bsubv):
    """Numba-accelerated RHS: -B_sub_theta / B_sub_phi evaluated spectrally."""
    B_sub_theta = 0.0
    B_sub_phi = 0.0
    for i in range(len(xm)):
        c = np.cos(xm[i] * theta - xn[i] * phi)
        B_sub_theta += bsubu[i] * c
        B_sub_phi += bsubv[i] * c
    return -B_sub_theta / B_sub_phi

def follow_contour_Phi(theta_0, phi_0, results, n_points = 1000, rtol=1e-10, atol=1e-12):
    """
    Follow a contour of Phi starting from the point (theta_0, phi_0) where Phi = Phi_0. This is done by integrating the equations:
        dphi/dtheta = -dPhi/dtheta / dPhi/dphi = -B_sub_theta / B_sub_phi
    Parameters
    ----------
    theta_0 : float
        The starting value of theta.
    phi_0 : float
        The starting value of phi.
    results : Struct
        The results of vmec_compute_geometry, which contains all the geometric quantities of interest.
    n_points : int
        The number of points to follow along the contour.
    rtol : float
        The relative tolerance for the ODE solver.
    atol : float
        The absolute tolerance for the ODE solver.
    Returns
    -------
    theta_contour : array
        The theta values along the contour.
    phi_contour : array
        The phi values along the contour.
    """
    #########################
    # SELECT KEY QUANTITIES #
    #########################
    if hasattr(results, "bsubumnc") and hasattr(results, "bsubvmnc"):
        # Magnetic field covariant components (include secular I, G via m=0,n=0 mode)
        bsubu = results.bsubumnc[0, :] if results.bsubumnc.ndim == 2 else results.bsubumnc  # shape (mnmax_nyq,)
        bsubv = results.bsubvmnc[0, :] if results.bsubvmnc.ndim == 2 else results.bsubvmnc  # shape (mnmax_nyq,)
        xm = results.xm_nyq
        xn = results.xn_nyq

    elif hasattr(results, "phimn"):
        # Check phimn 1D
        assert results.phimn.ndim == 1, "Expected phimn to be 1D array of spectral coefficients."

        # Check xm and xn
        if not hasattr(results, "xm_potential") or not hasattr(results, "xn_potential"):
            raise ValueError("Results must contain xm_potential and xn_potential for m=0,n=0 mode if using phimn.")

        # Check current potential
        if not hasattr(results, "G_current") or not hasattr(results, "I_current"):
            raise ValueError("Results must contain G_current and I_current for m=0,n=0 mode if using phimn.")

        # Get spectral mode labels
        xm = results.xm_potential
        xn = results.xn_potential

        # Get m=0, n=0 index
        m0n0_index = np.where((xm == 0) & (xn == 0))

        # Get current
        G_current = results.G_current
        I_current = results.I_current

        # Compute the partial derivatives of Phi
        bsubu = differentiate_spectral(results.phimn, xm, xn, s_or_c = "s", m_or_n = "m")    
        bsubu[m0n0_index] = I_current/2/np.pi
        bsubv = differentiate_spectral(results.phimn, xm, xn, s_or_c = "s", m_or_n = "n")
        bsubv[m0n0_index] = G_current/2/np.pi
    else:
        raise ValueError("Results must contain either bsubumnc and bsubvmnc, or phimn with xm_potential and xn_potential.")


    #####################
    # DEFINE THE ODE RHS #
    #####################
    # Cast to contiguous float64 for numba
    xm_f = np.ascontiguousarray(xm, dtype=np.float64)
    xn_f = np.ascontiguousarray(xn, dtype=np.float64)
    bsubu_f = np.ascontiguousarray(bsubu, dtype=np.float64)
    bsubv_f = np.ascontiguousarray(bsubv, dtype=np.float64)

    def dphi_dtheta(theta, phi):
        return _dphi_dtheta_rhs(theta, phi[0], xm_f, xn_f, bsubu_f, bsubv_f)

    #####################
    # INTEGRATE THE ODE #
    #####################
    theta_span = (theta_0, theta_0 + 2 * np.pi)
    t_eval = np.linspace(theta_span[0], theta_span[1], n_points)

    sol = solve_ivp(dphi_dtheta, theta_span, [phi_0],
                    t_eval=t_eval, rtol=rtol, atol=atol)

    theta_contour = sol.t
    phi_contour = sol.y[0]

    return theta_contour, phi_contour

def compute_Phi_values(results, theta, phi, secular = True):
    """
    Compute Phi on the grid using the digested VMEC surface.

    Parameters
    ----------
    results : Struct
        The results of vmec_compute_geometry, which contains all the geometric quantities of interest.
    theta : array
        The theta values of the grid, ordered as (ntheta,).
    phi : array
        The phi values of the grid, ordered as (nphi,).
    secular : bool
        Whether to include the secular contributions from G and I in the m=0,n=0 mode. Default is True.

    Returns
    -------
    Phi_grid : array
        The values of Phi on the grid, ordered as (ns, ntheta, nphi).
    """

    Phimns = compute_Phi(results)
    xm = results.xm_nyq
    xn = results.xn_nyq
    G_Boozer = results.G_Boozer
    I_Boozer = results.I_Boozer

    Phi_obj = PhiResults(Phimns, G_Boozer, I_Boozer, xm, xn, secular=secular)

    # Compute Phi on grid
    Phi_grid = Phi_obj.evaluate_on_grid(theta, phi)

    return Phi_grid

def _carve_single_coil(args):
    """
    Worker function for multiprocessing in make_coils_from_vmec.
    Processes a single coil contour.
    """
    j, phi_0, theta_start, data_vs, N_theta, full_info, no_B, no_excursion, I_val = args

    theta_c, phi_c = follow_contour_Phi(theta_start, phi_0, data_vs, n_points=N_theta)

    coil = {"theta": theta_c, "phi": phi_c}

    if full_info:
        R_c, Z_c, dR_c, dZ_c, dphi_c, d2R_c, d2Z_c, d2phi_c, d3R_c, d3Z_c, d3phi_c = \
            evaluate_contour_RZ(theta_c, phi_c, data_vs, full_output=True)
        
        _, _, normal_unit, gaussian_curvature, mean_curvature = \
            evaluate_surface_geometry(theta_c, phi_c, data_vs)
        
        curvature, torsion, normal_curvature, geodesic_curvature = characterise_curve(R_c, Z_c, phi_c, theta_c,
                                                dR=dR_c, dZ=dZ_c, dphi_dtheta=dphi_c,
                                                d2R=d2R_c, d2Z=d2Z_c, d2phi_dtheta=d2phi_c,
                                                d3R=d3R_c, d3Z=d3Z_c, d3phi_dtheta=d3phi_c,
                                                surface_normal=normal_unit)

        coil["R"] = R_c
        coil["Z"] = Z_c
        coil["curvature"] = curvature
        coil["torsion"] = torsion
        coil["gaussian_curvature"] = gaussian_curvature
        coil["mean_curvature"] = mean_curvature
        coil["normal_curvature"] = normal_curvature
        coil["geodesic_curvature"] = geodesic_curvature
        # Cartesian coordinates
        coil["X"] = R_c * np.cos(phi_c)
        coil["Y"] = R_c * np.sin(phi_c) 

        if not no_B:
            curvature_B, normal_unit_B = evaluate_curvature_B_field(theta_c, phi_c, data_vs)
            normal_curvature_B = curvature_B * np.sum(normal_unit * normal_unit_B, axis=-1)
            coil["curvature_B"] = curvature_B
            coil["normal_curvature_B"] = normal_curvature_B
        
        # Off-planar excursion
        if not no_excursion:
            max_exc, avrg_distance, std_distance = max_excursion(coil["X"], coil["Y"], coil["Z"])
            coil["max_excursion"] = max_exc
            coil["avrg_distance"] = avrg_distance
            coil["std_distance"] = std_distance
    else:
        R_c, Z_c = evaluate_contour_RZ(theta_c, phi_c, data_vs, full_output=False)
        coil["R"] = R_c
        coil["Z"] = Z_c
        coil["X"] = R_c * np.cos(phi_c)
        coil["Y"] = R_c * np.sin(phi_c)

    coil["current"] = I_val
    return j, coil
   
def make_coils_from_vmec(vmec_file, s, N_coils, N_theta = 100, full_info = False, no_B = False, no_excursion = False, parallel = False, Phi_data = None):
    """
    Given a VMEC file, extract the geometry and compute proxy coil shapes by following contours of Phi. This is done by:
        1) Loading the VMEC file
        2) Computing Phi
        3) Following contours of Phi
        4) Computing the R, Z, phi coordinates  and associated current
        5) Optionally characterising the curves

    Parameters
    ----------
    vmec_file : str
        The path to the VMEC file to load.
    s : float
        The normalized flux surface to extract.
    N_coils : int
        The number of coils (contours) to follow.
    full_info : bool
        If True, then also characterise the curves and plot them in 3D. If False, then just return the coil shapes as arrays of R, Z, phi.
    no_B : bool
        If True, then skip the evaluation of the magnetic field and related quantities, which can be time-consuming. This is only relevant if full_info is True.
    no_excursion : bool
        If True, then skip the evaluation of the off-planar excursion, which can be time-consuming. This is only relevant if full_info is True.
    parallel : bool
        If True, then follow the contours in parallel using multiprocessing. If False, then follow the contours sequentially. Note that parallel=True can be much faster for large N_coils, but may cause issues with memory if the VMEC data is large, since it needs to be copied to each process.

    Returns
    -------
    coils : list of tuples
        A list of length N_coils, where each element is a tuple (R_c, Z_c, phi_c) containing the R, Z and phi values along that coil contour.
    data_vs : Struct
        The data from vmec_compute_geometry, which contains all the geometric quantities of interest that were used to compute the coils. This is returned for convenience so that it can be used for plotting or further analysis without needing to recompute the geometry.
    """
    if Phi_data == None:
        #############
        # LOAD VMEC #
        #############
        # Load the VMEC file and compute geometry
        if isinstance(vmec_file, str):
            vs = load_vmec(vmec_file)
        else:
            vs = vmec_file

        # Extract data
        data_vs = vmec_compute_geometry(vs, s, None, None, full_output=False)

        ###############
        # COMPUTE PHI #
        ###############
        # Compute Phi 
        Phimns = compute_Phi(data_vs)

    elif hasattr(Phi_data, 'contents'):
        # Check if all the attributes in contents are present
        for attr in Phi_data.contents:
            if not hasattr(Phi_data, attr):
                raise ValueError(f"Phi_data is missing required attribute '{attr}'.")
        data_vs = copy.copy(Phi_data)
        mu0 = 4*np.pi*1e-7
        Phimns = Phi_data.phimns * mu0
        data_vs.G_Boozer = Phi_data.G_current * mu0 / (2*np.pi)   # This is how Phi is normalised from REGCOIL
        data_vs.I_Boozer = Phi_data.I_current * mu0 / (2*np.pi)   # so that Phi = ... + I_current / (2π) * theta + G_current / (2π) * phi + ...

    else:
        raise ValueError("Phi_data must be either None or a Struct containing phimns, G_current, I_current, xm_nyq and xn_nyq.")

    Phi_obj = PhiResults(Phimns, data_vs.G_Boozer, data_vs.I_Boozer,
                data_vs.xm_nyq, data_vs.xn_nyq)

    #############
    # CUT COILS #
    #############
    # Cut coils at different phi values
    theta_start = 0.0
    phi_starts = np.linspace(0, 2 * np.pi, N_coils, endpoint=False)

    # Phi values at the midpoints of the contours, used to estimate coil currents
    phi_ext = np.append(phi_starts, phi_starts[0] + 2 * np.pi)
    Phi_vals = Phi_obj.evaluate_on_grid(np.array([0]), phi_ext)
    Phi_vals = np.squeeze(Phi_vals)  # shape (N_coils,)
    Phi_vals_mid = 0.5 * (Phi_vals[1:] + Phi_vals[:-1])
    I_vals = [Phi_vals_mid[i] - Phi_vals_mid[i-1] for i in range(1, len(Phi_vals_mid))]
    I_0 = Phi_vals_mid[0] - Phi_vals[0] - Phi_vals_mid[-1] + Phi_vals[-1]
    I_vals = np.insert(I_vals, 0, I_0)

    # Carve coils
    coils = {}
    if parallel:
        args_list = [(j, phi_0, theta_start, data_vs, N_theta, full_info, no_B, no_excursion, I_vals[j])
                     for j, phi_0 in enumerate(phi_starts)]
        with multiprocessing.Pool() as pool:
            results = pool.map(_carve_single_coil, args_list)
        for j, coil in results:
            coils[j] = coil
    else:
        for j, phi_0 in enumerate(phi_starts):
            # Compute contour of Phi starting from (theta_start, phi_0)
            theta_c, phi_c = follow_contour_Phi(theta_start, phi_0, data_vs, n_points=N_theta)

            # Save the contour
            coils[j] = {"theta": theta_c, "phi": phi_c}

            if full_info:
                R_c, Z_c, dR_c, dZ_c, dphi_c, d2R_c, d2Z_c, d2phi_c, d3R_c, d3Z_c, d3phi_c = \
                    evaluate_contour_RZ(theta_c, phi_c, data_vs, full_output=True)
                
                _, _, normal_unit, gaussian_curvature, mean_curvature = \
                    evaluate_surface_geometry(theta_c, phi_c, data_vs)
                
                curvature, torsion, normal_curvature, geodesic_curvature = characterise_curve(R_c, Z_c, phi_c, theta_c,
                                                        dR=dR_c, dZ=dZ_c, dphi_dtheta=dphi_c,
                                                        d2R=d2R_c, d2Z=d2Z_c, d2phi_dtheta=d2phi_c,
                                                        d3R=d3R_c, d3Z=d3Z_c, d3phi_dtheta=d3phi_c, 
                                                        surface_normal=normal_unit)

                coils[j]["R"] = R_c
                coils[j]["Z"] = Z_c
                coils[j]["curvature"] = curvature
                coils[j]["torsion"] = torsion
                coils[j]["normal_curvature"] = normal_curvature
                coils[j]["geodesic_curvature"] = geodesic_curvature
                coils[j]["gaussian_curvature"] = gaussian_curvature
                coils[j]["mean_curvature"] = mean_curvature

                # Cartesian coordinates
                coils[j]["X"] = R_c * np.cos(phi_c)
                coils[j]["Y"] = R_c * np.sin(phi_c) 

                # Evaluate curvature of B field and normal curvature in B field direction
                if not no_B:
                    curvature_B, normal_unit_B = evaluate_curvature_B_field(theta_c, phi_c, data_vs)
                    normal_curvature_B = curvature_B * np.sum(normal_unit * normal_unit_B, axis=-1)
                    coils[j]["curvature_B"] = curvature_B
                    coils[j]["normal_curvature_B"] = normal_curvature_B

                # Off-planar excursion
                if not no_excursion:
                    max_exc, avrg_distance, std_distance = max_excursion(coils[j]["X"], coils[j]["Y"], coils[j]["Z"])
                    coils[j]["max_excursion"] = max_exc
                    coils[j]["avrg_distance"] = avrg_distance
                    coils[j]["std_distance"] = std_distance


            else:
                R_c, Z_c = evaluate_contour_RZ(theta_c, phi_c, data_vs, full_output=False)
                coils[j]["R"] = R_c
                coils[j]["Z"] = Z_c

                # Cartesian coordinates
                coils[j]["X"] = R_c * np.cos(phi_c)
                coils[j]["Y"] = R_c * np.sin(phi_c)
            
            # Compute the current associated to this coil
            coils[j]["current"] = I_vals[j]

    return coils, data_vs

def max_excursion(X, Y, Z, verbose = False):
    """
    Compute the maximum excursion of a curve in 3D space, defined as the minimum max-distance from any point to a plane. 
    This is a measure of how "planar" the curve is, with smaller values indicating a more planar curve.

    Parameters
    ----------
    X : array
        The X values of the curve, ordered as (n_points,).
    Y : array
        The Y values of the curve, ordered as (n_points,).
    Z : array
        The Z values of the curve, ordered as (n_points,).
    verbose : bool
        If True, print additional information about the plane and distances. If False, only return the maximum excursion.

    Returns
    -------
    max_excursion : float
        The maximum excursion of the curve, defined as the minimum max-distance from any point to a plane.
    """
    # Use SVD to find the best-fit plane to the points (X, Y, Z)
    points = np.stack((X, Y, Z), axis=-1)  # shape (n_points, 3)
    points_mean = np.mean(points, axis=0)
    points_centered = points - points_mean
    U, S, Vt = np.linalg.svd(points_centered)
    normal_vector = Vt[2, :]  # The normal vector to the best-fit plane is the last row of Vt

    # Compute the distance from each point to the plane defined by the normal vector and the mean point
    dist = points - points_mean
    distances = np.abs(np.dot(dist, normal_vector)) / np.linalg.norm(normal_vector)
    distances_tangent = np.linalg.norm(dist - np.outer(np.dot(dist, normal_vector) / np.linalg.norm(normal_vector)**2, normal_vector), axis=1)
    max_excursion = np.max(distances)
    avrg_distance = np.mean(distances_tangent)

    # Measure degree of circularity in the plane
    std_distance = np.max([distances_tangent.max() - avrg_distance, avrg_distance - distances_tangent.min()])  # Avoid division by zero

    if verbose:
        print(f"Best-fit plane normal vector: {normal_vector}")
        print(f"Mean point on plane: {points_mean}")
        print(f"Distances from points to plane: {distances}")
        print(f"Maximum excursion (max distance to plane): {max_excursion}")
        print(f"Average distance in plane: {avrg_distance}")
        print(f"Max deviation on plane: {std_distance}")
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X, Y, Z, color='k')
        # Plot the best-fit plane
        plane_size = 1.5 * np.max(np.linalg.norm(points_centered, axis=1))
        plane_x, plane_y = np.meshgrid(np.linspace(-plane_size, plane_size, 10), np.linspace(-plane_size, plane_size, 10))
        plane_z = (-normal_vector[0] * plane_x - normal_vector[1] * plane_y) / normal_vector[2]
        ax.plot_surface(points_mean[0] + plane_x, points_mean[1] + plane_y, points_mean[2] + plane_z, color='lightgray', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Max Excursion: {max_excursion:.3e}')
        plt.show()

    return max_excursion, avrg_distance, std_distance

def plot_coils(coils, vs_data = None, ax=None, **kwargs):
    """
    Given a dictionary of coils (as returned by make_coils_from_vmec), plot the coil shapes in 3D.

    Parameters
    ----------
    coils : dict
        A dictionary where each key is a coil index and each value is a dictionary containing the coil data, including "R", "Z" and "phi" arrays.
    vs_data : Struct, optional
        The data from vmec_compute_geometry, which can be used to plot the toroidal surface from the VMEC geometry. If None, then only the coils are plotted.
    ax : matplotlib 3D axis, optional
        If provided, the coils will be plotted on this axis. If None, then a new figure and axis will be created.
    kwargs : dict
        Additional keyword arguments to pass to the ax.plot function when plotting the coils, such as linewidth or alpha.
    """
    #################
    # CREATE FIGURE #
    #################
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    #####################
    # PLOT VMEC SURFACE #
    #####################
    if vs_data is not None:
        # Theta and phi grids
        theta_surf = np.linspace(0, 2 * np.pi, 50)
        phi_surf = np.linspace(0, 2 * np.pi, 50)
        Theta_surf, Phi_surf = np.meshgrid(theta_surf, phi_surf, indexing='ij')

        # Evaluate R, Z on the surface using the spectral coefficients
        angle = vs_data.xm[:, None, None] * Theta_surf[None, :, :] - vs_data.xn[:, None, None] * Phi_surf[None, :, :]
        R_surf = np.einsum('i,ijk->jk', vs_data.rmnc[0, :], np.cos(angle))
        Z_surf = np.einsum('i,ijk->jk', vs_data.zmns[0, :], np.sin(angle))
        X_surf = R_surf * np.cos(Phi_surf)
        Y_surf = R_surf * np.sin(Phi_surf)

        # Plot the surface
        ax.plot_surface(X_surf, Y_surf, Z_surf, color='lightgray', alpha=0.2, rstride=1, cstride=1, edgecolor='none')

    ####################
    # PLOT COIL CURVES #
    ####################
    # Define colors for coils depending on current
    currents = np.array([coils[j]["current"] for j in coils])
    norm = plt.Normalize(vmin=np.min(currents), vmax=np.max(currents))
    cmap = plt.get_cmap('plasma')

    # Plot each coil in 3D
    for j in coils:
        R_c = coils[j]["R"]
        Z_c = coils[j]["Z"]
        phi_c = coils[j]["phi"]
        X_c = R_c * np.cos(phi_c)
        Y_c = R_c * np.sin(phi_c)
        ax.plot(X_c, Y_c, Z_c, color=cmap(norm(coils[j]["current"])), linewidth=2, **kwargs)

    # Add colorbar for currents
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.ax.set_title(r'$I$')

    #################
    # FINALIZE PLOT #
    #################
    ax.set_aspect('equal')
    
    # Eliminate all axes
    ax.axis('off')

    # Clean up the grid and panes
    ax.grid(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

def analyse_coils():
    ###############
    # MAKE COILS #
    ##############
    vmec_file = "/home/erodrigu/Python/configs/wout_precise_QH.nc"
    s = 1.0
    N_coils = 100
    coils, data_vs = make_coils_from_vmec(vmec_file, s, N_coils, N_theta=500, full_info=True, parallel=False)

    #################################
    # CHOOSE LARGEST CURVATURE COIL #
    #################################
    # Select largest curvature coil
    max_curvature = 0.0
    max_curvature_coil = None
    for j in coils:
        curvature = coils[j]["curvature"]
        max_curv_j = np.max(curvature)
        if max_curv_j > max_curvature:
            max_curvature = max_curv_j
            max_curvature_coil = j
    
    ###################
    # 3D PLOT OF COIL #
    ###################
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_coils({max_curvature_coil: coils[max_curvature_coil]}, vs_data=data_vs, ax=ax)

    ###################
    # PLOT PROPERTIES #
    ###################
    coil = coils[max_curvature_coil]
    theta_c = coil["theta"]
    curvature = coil["curvature"]
    torsion = coil["torsion"]
    normal_curvature = coil["normal_curvature"]
    geodesic_curvature = coil["geodesic_curvature"]
    mean_curvature = coil["mean_curvature"]
    curvature_B = coil["curvature_B"]
    normal_curvature_B = coil["normal_curvature_B"]
    
    # Principal curvatures
    k1 = mean_curvature + np.sqrt(mean_curvature**2 - coil["gaussian_curvature"])
    k2 = mean_curvature - np.sqrt(mean_curvature**2 - coil["gaussian_curvature"])

    # Check kn_B + kn_coil = 2 mean_curvature
    plt.figure()
    plt.plot(theta_c, normal_curvature_B + normal_curvature, 'k-', label='$\\kappa_{n,B} + \\kappa_n$')
    plt.plot(theta_c, normal_curvature_B - normal_curvature, 'k-', label='$\\kappa_{n,B} - \\kappa_n$')
    plt.plot(theta_c, 2 * mean_curvature, 'r--', label='$2H$')
    plt.xlabel('$\\theta$')
    plt.ylabel('Curvature')
    plt.legend()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(theta_c, curvature, 'k-', label='$\\kappa$')
    ax1.plot(theta_c, np.abs(normal_curvature), 'k--', label='$\\kappa_n$')
    ax1.plot(theta_c, np.abs(geodesic_curvature), 'k:', label='$\\kappa_g$')
    assert np.allclose(curvature, np.sqrt(normal_curvature**2 + geodesic_curvature**2), rtol=1e-5, atol=1e-8), "Curvature does not match sqrt(kappa_n^2 + kappa_g^2)"
    # ax1.plot(theta_c, np.abs(mean_curvature), 'k-.', label='$H$')
    ax1.plot(theta_c, np.abs(k1), 'r:', label='$k_1$')
    ax1.plot(theta_c, np.abs(k2), 'b:', label='$k_2$')
    ax1.set_yscale('log')
    ax1.set_xlabel('$\\theta$')
    ax1.set_ylabel('Curvature')
    ax1.legend()
    ax2.plot(theta_c, torsion, 'k-')
    ax2.set_yscale('symlog')
    ax2.set_xlabel('$\\theta$')
    ax2.set_ylabel('Torsion')
    plt.tight_layout()
    plt.show()

def scan_excursion_s(vmec_file = "/home/erodrigu/Python/configs/wout_preciseQA.nc", folder = "./", plot = False):
    """
    Scan the maximum excursion of the coil contours as a function of the flux surface s.
    """
    s_values = np.linspace(0.1, 1.0, 20)
    max_excursions = []
    max_curvatures = []
    max_geo_curvatures = []
    with tqdm(total=len(s_values), desc="Scanning excursion across s") as pbar:
        for s in s_values:
            coils, data_vs = make_coils_from_vmec(vmec_file, s, N_coils=60, N_theta=100, full_info=True, parallel=False)
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
            print(max_geo_curvatures)

            print(f"s={s:.2f}, max excursion: {max_excursions[-1]:.3e}, max curvature: {max_curvatures[-1]:.3e}, max geodesic curvature: {max_geo_curvatures[-1]:.3e}")
            pbar.update(1)
        
    # Save results to file
    import pickle
    with open(folder + "coil_excursion_results_vmec.pkl", "wb") as f:
        pickle.dump({
            "s_values": s_values,
            "max_excursions": max_excursions,
            "max_curvatures": max_curvatures,
            "max_geo_curvatures": max_geo_curvatures
        }, f)

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(s_values, max_excursions, 'ko-', label='Max Excursion')
        plt.ylabel('Max Excursion')
        ax2 = plt.gca().twinx()
        # ax2.plot(s_values, np.array(max_curvatures), 'ko:', label='Max Curvature')
        ax2.plot(s_values, np.array(max_geo_curvatures), 'ks--', label='Max Geodesic Curvature')
        ax2.set_ylabel('Max Curvature')
        plt.xlabel('s')
        plt.tight_layout()
        plt.show()

def make_nice_3d_plot_coils(vmec_file=None, output_file="coil_set_k1.png", s_plot=0.9, n_coils=40):
    if vmec_file is None:
        vmec_file = os.environ.get("VMEC_WOUT_FILE")
    if vmec_file is None:
        raise ValueError("vmec_file must be provided or VMEC_WOUT_FILE must be set.")

    vs = load_vmec(vmec_file)
    s = [s_plot]
    N_theta = 100
    N_phi = 200
    theta = np.linspace(0, 2 * np.pi, N_theta)
    phi = np.linspace(0, 2 * np.pi, N_phi)
    results = vmec_compute_geometry(vs, s, theta, phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    R, Z, normal_unit, gaussian_curvature, mean_curvature = evaluate_surface_geometry(T, P, results)
    curvature_B, normal_unit_B = evaluate_curvature_B_field(T, P, results)
    normal_curvature_B = curvature_B * np.sum(normal_unit * normal_unit_B, axis=-1)
    R, Z, dR, dZ, dphi, d2R, d2Z, d2phi, d3R, d3Z, d3phi = evaluate_contour_RZ(T, P, results)
    curvature_coil, torsion_coil, normal_curvature_coil, geodesic_curvature_coil =  characterise_curve(R, Z, P, T, dR, dZ, dphi, d2R, d2Z, d2phi, d3R, d3Z, d3phi, surface_normal=normal_unit)
    k1 = mean_curvature + np.sqrt(mean_curvature**2 - gaussian_curvature)
    k2 = mean_curvature - np.sqrt(mean_curvature**2 - gaussian_curvature)

    # Plot the surface colored by mean curvature, with hover showing the mean curvature value

    # Main surface (no colorbar)
    fig = go.Figure(data=[
        go.Surface(
            x=R*np.cos(P),
            y=R*np.sin(P),
            z=Z,
            surfacecolor=k1,
            customdata=k1,  
            colorscale='Gray_r',
            cmin=0,
            cmax=np.max(np.abs(k1)),
            # Disable lighting so grayscale maps directly to k1 values.
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0, roughness=1.0),
            showscale=True,
            colorbar=dict(
            title={
                'text': 'κ₁',
                'side': 'right',
                'font': dict(size=18)
            },
            thickness=40,
            len=0.5,
            x=0.76,
            y=0.47,
            outlinewidth=3,
            showticklabels=False,
        ),

        )
    ])

    # Add the coils as 3D lines on top of the surface
    N_coils = n_coils
    # Make coils from vmec
    coils, data_vs = make_coils_from_vmec(vmec_file, 1.0, N_coils, N_theta=300, full_info=False, parallel=False)
    # Plot the coils
    for coil_id in coils:
        R_coil = np.squeeze(coils[coil_id]['R'])
        Z_coil = np.squeeze(coils[coil_id]['Z'])
        phi_coil = np.squeeze(coils[coil_id]['phi'])
        x_coil = R_coil * np.cos(phi_coil)
        y_coil = R_coil * np.sin(phi_coil)
        z_coil = Z_coil
        fig.add_trace(go.Scatter3d(
            x=x_coil,
            y=y_coil,
            z=z_coil,
            mode='lines',
            line=dict(color='black', width=13),
            showlegend=False
        ))

    # Eliminate background and axes for better visualization, and lock aspect ratio
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'  # lock aspect ratio to data units
        ),
        autosize=True,
        # title='Principal Curvature and Coils'
    )

    # Save the final image
    fig.write_image(output_file, width=1600, height=1000, scale=2.0)

    # fig.show()

