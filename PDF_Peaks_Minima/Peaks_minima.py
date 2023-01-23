import numpy as np
import healpy as hp

def find_extrema(kappa_map,minima=False,lonlat=False):
    """find extrema in a smoothed masked healpix map
       default is to find peaks, finds minima with minima=True
    
       Parameters
       ----------
       kappa_masked_smooth: MaskedArray (healpy object)
           smoothed masked healpix map for which extrema are to be identified
       minima: bool
           if False, find peaks. if True, find minima
       
       Returns
       -------
       extrema_pos: np.ndarray
           extrema positions on sphere, theta and phi, in radians
       extrema_amp: np.ndarray
           extrema amplitudes in kappa
       
    """

    #first create an array of all neighbours for all valid healsparse pixels
    nside = hp.get_nside(kappa_map) #get nside
    ipix = np.arange(hp.nside2npix(nside))[kappa_map.mask==False] #list all pixels and remove masked ones
    neighbours = hp.get_all_neighbours(nside, ipix) #find neighbours for all pixels we care about

    #get kappa values for each pixel in the neighbour array
    neighbour_vals = kappa_map.data[neighbours.T]
    #get kappa values for all valid healsparse pixels
    pixel_val = kappa_map.data[ipix]

    #compare all valid healsparse pixels with their neighbours to find extrema
    if minima:
        extrema = np.all(np.tile(pixel_val,[8,1]).T < neighbour_vals,axis=-1)
    else:
        extrema = np.all(np.tile(pixel_val,[8,1]).T > neighbour_vals,axis=-1)

        
    #print the number of extrema identified
    if minima:
        print(f'number of minima identified: {np.where(extrema)[0].shape[0]}')
    else:
        print(f'number of peaks identified: {np.where(extrema)[0].shape[0]}')

    extrema_pos = np.asarray(hp.pix2ang(nside, ipix[extrema],lonlat=lonlat)).T #find the extrema positions
    extrema_amp = kappa_map[ipix][extrema].data #find the extrema amplitudes
    
    return extrema_pos, extrema_amp