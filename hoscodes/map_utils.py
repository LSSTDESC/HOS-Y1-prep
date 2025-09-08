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


def smoothing(raw_map,scale_length):
    """
    map smoothing in healpy applied to self.mapbins in class.
    -----------
    parameters:
    sl: smoothing lenght in arcmins.
    
    """
    
    scale_length_radians = sl/60/180*np.pi
    map_healpy_mask = hp.ma(raw_map) #applies healpy masked object 

    
    return hp.smoothing(map_healpy_mask,sigma = scale_length_radians) 


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
