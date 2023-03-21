#cd ~/HOS-Y1-prep/hoscodes/
import numpy as np
import healpy as hp
from glob import glob
from HOScodes import kappamaps
import os,sys

sl_arcmin=10.25 #define the smoothing length for the map in arcmins 
nside= 4096# We down the original resolution (NSIDE=8192) to the one used by the other maps
#map2 nside
nshells = [19,106] #
seeds=[1,2]
nzs='kappa_Euclid_dndz_fu08_bin1-5'

#raw_maps_path = '/global/cscratch1/sd/jatorres/KappaMapsLR/SkySim5000/'
raw_maps_path = '/global/cscratch1/sd/xuod/HOS_sims/L845/HACC150/'

#filenames = [raw_maps_path+'SkySim5000_'+nzs+'_kappa_tomo%d.fits'%l for l in range(1,6)]
for nshell in nshells:
    for seed in seeds:
        filenames = [os.path.join(raw_maps_path,                          f'shells_z{nshell}_subsampleauto_groupiso/{nzs}/kappa_hacc_nz%d_nside4096_seed{seed}.fits'%l) for l in range(5)]
        kappa_maps_field = kappamaps(filenames=filenames,nshells=nshell,seed=seed,nzs=nzs,nside=nside)
        kappa_maps_field.readmaps_fits()

        dir_smoorhed_maps = os.path.join(os.getcwd(),'smoothed/')
        if not os.path.exists(dir_smoorhed_maps):
            os.makedirs(dir_smoorhed_maps)

        for i in range(kappa_maps_field.Nbins):
            fname = kappa_maps_field.filenames[i]
            kappa_map = kappa_maps_field.mapbins[i]
            smoothed_map = kappa_maps_field.smoothing_maps(kappa_map,sl_arcmin)
            wname = 'smoothed_map_'+fname.split('/')[-1]
            hp.write_map(dir_smoorhed_maps+wname.split('.')[0]+'_nshell%d_.fits'%nshell,smoothed_map)