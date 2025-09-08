import numpy as np
import os,sys
import healpy as hp
from glob import glob
from astropy.io import fits

def RaDec2pixel(ra, dec, nside):
    #converts ra dec to healpy pixel index 
    #assumes that input ra and dec are in radians
    return hp.pixelfunc.ang2pix(nside, np.pi/2.0 + np.radians(-dec), np.radians(ra))

def create_healpy_map(ra, dec, g1, g2, nside):
    
    npix = hp.nside2npix(nside)
    g1_map = np.zeros(npix)
    g2_map = np.zeros(npix)
    w_map = np.zeros(npix)
    footprint_map = np.zeros(npix)
    
    pix = RaDec2pixel(ra,dec,nside=nside)
    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    w_map[unique_pix] += np.bincount(idx_rep, weights=w)
    g1_map[unique_pix] += np.bincount(idx_rep, weights=g1*w)
    g2_map[unique_pix] += np.bincount(idx_rep, weights=g2*w)
    mask_map = w_map != 0. # this is footprint of the map (in this bin)
    
    footprint_map[mask_map] = 1
    g1_map[mask_map] =  g1_map[mask_map] / w_map[mask_map]  
    g2_map[mask_map] =  g2_map[mask_map] / w_map[mask_map]  
    
    return g1_map, g2_map, w_map, footprint_map

catpath = sys.argv[1]
model = catpath.split('/')[-2]
p = sys.argv[2]
n_g = '0.6'
#catpath = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/'+IA_model+'/'
NSIDE = 2048

filepath_map_data = '/pscratch/sd/j/jatorres/data/GalCat/SkySim5000IA/'+model +'/'# the healpy maps will be stored in this path

if (os.path.isdir(filepath_map_data) == False):
    os.makedirs(filepath_map_data)

filenames = [catpath+'galcat_skysim5000_'+p+'_tomo%d.dat'%i for i in range(1,6)]

for i,l in enumerate(filenames):
    tomo_bin = 'tomo'+str(i+1)
    
    cat = np.loadtxt(l,skiprows=1,usecols=(0,1,2,3,4,9,10,5))
    #cat = np.loadtxt(l,skiprows=1,usecols=(0,1,2,3,4,5))
    ra, dec, z, g1, g2, g1_IA, g2_IA, w = cat.T

    g1_map, g2_map, w_map, footprint_map = create_healpy_map(ra, dec, g1, g2, NSIDE)
    #IAg1_map, IAg2_map, _, _ = create_healpy_map(ra, dec, g1_IA, g2_IA, NSIDE)


    hp.write_map(filepath_map_data+'g1_GalMap_'+IA_model+'_'+p+'_'+tomo_bin+'_All_'+n_g+'GpAM_nside_'+str(NSIDE)+'.fits', g1_map, overwrite=True)
    hp.write_map(filepath_map_data+'g2_GalMap_'+IA_model+'_'+p+'_'+tomo_bin+'_All_'+n_g+'GpAM_nside_'+str(NSIDE)+'.fits', g2_map, overwrite=True)
    #hp.write_map(filepath_map_data+'g1+IA_GalMap_'+IA_model+'_'+p+'_'+tomo_bin+'_All_'+n_g+'GpAM_nside_'+str(NSIDE)+'.fits', IAg1_map, overwrite=True)
    #hp.write_map(filepath_map_data+'g2+IA_GalMap_'+IA_model+'_'+p+'_'+tomo_bin+'_All_'+n_g+'GpAM_nside_'+str(NSIDE)+'.fits', IAg2_map, overwrite=True)
    hp.write_map(filepath_map_data+'footprint_'+IA_model+'_'+p+'_'+tomo_bin+'_All_'+n_g+'GpAM_nside_'+str(NSIDE)+'.fits', footprint_map, overwrite=True)
    hp.write_map(filepath_map_data+'weight_'+IA_model+'_'+p+'_'+tomo_bin+'_All_'+n_g+'GpAM_nside_'+str(NSIDE)+'.fits', w_map, overwrite=True)