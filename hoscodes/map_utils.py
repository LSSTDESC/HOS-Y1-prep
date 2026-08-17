import numpy as np
import healpy as hp
from healpy.pixelfunc import ud_grade
import os,sys

#methods to deal with individual maps

def fits_readmap(filename):

    nside_c=32 # nside for creating patches don't change

    fits_map_healsparse = healsparse.HealSparseMap.read(filename,nside_c)    
    
    return fit_map_healsparse
    
def numpy_readmap(filename):
    numpy_map = np.load(filename)

    return numpy_map
    
def healpy_readmap(filename):
    healpy_map = hp.read_map(filename) 

    return healpy_map


def infer_lmax(scale_length, nside, beam_tolerance=1e-4):
    """
    Infer lmax for Gaussian smoothing.

    Parameters
    ----------
    scale_length : float
        Gaussian FWHM in arcminutes.
    nside : int
        HEALPix NSIDE.
    beam_tolerance : float, optional
        Stop retaining multipoles once the Gaussian beam response falls below
        this value. Default is 1e-4.

    Returns
    -------
    int
        Recommended lmax, capped at 3*nside - 1.

    Notes
    -----
    The Gaussian harmonic response is

        B_l = exp[-0.5 * l * (l + 1) * sigma**2]

    where sigma is the Gaussian width in radians.
    """
    scale_length = float(scale_length)
    nside = int(nside)
    beam_tolerance = float(beam_tolerance)

    if not np.isfinite(scale_length) or scale_length <= 0:
        raise ValueError("scale_length must be positive.")

    if not hp.isnsideok(nside):
        raise ValueError(f"{nside} is not a valid HEALPix NSIDE.")

    if not 0 < beam_tolerance < 1:
        raise ValueError("beam_tolerance must lie between 0 and 1.")

    fwhm_radians = np.deg2rad(scale_length / 60.0)
    sigma_radians = fwhm_radians / np.sqrt(8.0 * np.log(2.0))

    # Solve:
    # tolerance = exp[-0.5*l*(l+1)*sigma**2]
    argument = (
        1.0
        - 8.0 * np.log(beam_tolerance) / sigma_radians**2
    )

    beam_lmax = int(np.ceil(
        (-1.0 + np.sqrt(argument)) / 2.0
    ))

    pixel_lmax = 3 * nside - 1

    return min(beam_lmax, pixel_lmax)


def smoothing(raw_map, footprint_mask, scale_length):
    """
    Smooth a masked kappa map with boundary normalization.

    Parameters
    ----------
    raw_map : array-like
        Full HEALPix kappa map.
    footprint_mask : array-like
        Binary footprint: 1 for valid pixels and 0 outside.
    scale_length : float
        Gaussian FWHM in arcminutes.

    Returns
    -------
    healpy masked array
        Boundary-corrected smoothed map with the footprint applied.
    """
    kappa = np.asarray(raw_map, dtype=np.float64)
    footprint = np.asarray(footprint_mask)

    if kappa.ndim != 1 or footprint.ndim != 1:
        raise ValueError("raw_map and footprint_mask must be 1D.")

    if kappa.shape != footprint.shape:
        raise ValueError(
            "raw_map and footprint_mask must have the same shape."
        )

    nside = hp.get_nside(kappa)
    scale_length = float(scale_length)

    valid = (
        (footprint > 0)
        & np.isfinite(kappa)
        & (kappa > -1.0e29)
    )

    weights = valid.astype(np.float64)

    # Outside values, including the -1.7e30 sentinel, are ignored.
    weighted_kappa = np.zeros_like(kappa, dtype=np.float64)
    weighted_kappa[valid] = kappa[valid]

    fwhm_radians = np.deg2rad(scale_length / 60.0)

    lmax = infer_lmax(
        scale_length=scale_length,
        nside=nside,
        beam_tolerance=1e-4,
    )

    options = {
        "fwhm": fwhm_radians,
        "lmax": lmax,
        "iter": 0,
        "pol": False,
    }

    smooth_kappa = hp.smoothing(
        weighted_kappa,
        **options,
    )

    smooth_weights = hp.smoothing(
        weights,
        **options,
    )

    minimum_weight = 0.05
    usable = valid & (smooth_weights > minimum_weight)

    corrected = np.full(kappa.shape, hp.UNSEEN, dtype=np.float64)

    np.divide(
        smooth_kappa,
        smooth_weights,
        out=corrected,
        where=usable,
    )

    result = hp.ma(corrected)
    result.mask = ~usable

    return result

def gamma_read_maps(g1_name,g2_name,w_name,footprint_name,p):
    
    g1_map = healpy_readmap(g1_name)
    g2_map = healpy_readmap(g2_name)
    w_map = healpy_readmap(w_name)
    footprint_map = healpy_readmap(footprint_name)
    
    return g1_map, g2_map, w_map, footprint_map

def pixel2RaDec(index, nside, nest=False):
    #converts healpy pixel index to ra dec (in degrees)
    theta, phi = hp.pixelfunc.pix2ang(nside ,index,nest=nest)
    return np.degrees(phi), -np.degrees(theta-np.pi/2.0)

def RaDec2pixel(ra, dec, nside):
    #converts ra dec to healpy pixel index 
    #assumes that input ra and dec are in radians
    return hp.pixelfunc.ang2pix(nside, np.pi/2.0 + np.radians(-dec), np.radians(ra))

def Q_T(theta, theta_Q):
    # eqn (3.2) of http://articles.adsabs.harvard.edu/pdf/2005IAUS..225...81K

    return theta**2/(4*np.pi*theta_Q**4)*np.exp(-theta**2/(2*theta_Q**2))

def calculate_patch_radius(patch_area_sq_degrees):
    return math.acos(1-patch_area_sq_degrees*np.pi/(2*180*180))
