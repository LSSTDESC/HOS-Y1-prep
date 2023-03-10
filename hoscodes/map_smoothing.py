#cd ~/HOS-Y1-prep/hoscodes/
import numpy as np
import healpy as hp
from glob import glob
from HOScodes import kappamaps
import os,sys

raw_maps_path = '/global/cscratch1/sd/jatorres/KappaMapsLR/SkySim5000/'
sl_arcmin=10.25 #define the smoothing length for the map in arcmins 
nside= 4096# We down the original resolution (NSIDE=8192) to the one used by the other maps
#map2 nside
nshells = 57
seed=0
nzs='Euclid_dndz_fu08_bin1-5'
filenames = [raw_maps_path+'SkySim5000_'+nzs+'.dat_kappa_tomo%d.fits'%l for l in range(1,6)]

kappa_maps_SkySim5000 = kappamaps(filenames=filenames,nshells=nshells,seed=seed,nzs=nzs,nside=nside)
kappa_maps_SkySim5000.readmaps_healpy()

dir_smoorhed_maps = os.path.join(os.getcwd(),'smoothed/')
if not os.path.exists(dir_smoorhed_maps):
    os.makedirs(dir_smoorhed_maps)

for i in range(kappa_maps_SkySim5000.Nbins):
    fname = kappa_maps_SkySim5000.filenames[i]
    kappa_map = kappa_maps_SkySim5000.mapbins[i]
    smoothed_map = kappa_maps_SkySim5000.smoothing_maps(kappa_map,sl_arcmin)
    wname = 'smoothed_map_'+fname.split('/')[-1]
    hp.write_map(wname,smoothed_map)