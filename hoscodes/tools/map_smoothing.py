#cd ~/HOS-Y1-prep/hoscodes/
import numpy as np
import healpy as hp
from glob import glob
from hoscodes.map_utils import *
from hoscodes import WLmaps
import os,sys

kappa_maps_directory = sys.argv[1]



ia_model = 'NLA'#maps_names.split('/')[-2] 
noise = 'noisefree'

filenames = [kappa_maps_directory+'kappa_skysim5000_'+ia_model+'_'+noise+'_tomo%d.dat.npy'%i for i in range(1,6)]

theta_smoothing_scales = [2,5,10,15] #define the smoothing length for the map in arcmins 
nside=4096# We down the original resolution (NSIDE=8192) to the one used by the other maps

output_map_dir = '/pscratch/sd/j/jatorres/KappaMaps/Smoothed_Maps/HACC-Y1/'+IA_model+'/'

if (os.path.isdir(output_map_dir) == False):
    os.makedirs(output_map_dir)


all_sky_map = np.load('/global/homes/j/jatorres/misc/mask_allsky.npy')

void_val = -1.6375000e+30
void_map = np.full(nside**2*12,void_val)
map_mask_buffer =  void_map.copy()

kappa_maps = kappacodes(dir_results,filenames,nside)
kappa_maps.readmaps_npy()
for s_i,s in enumerate(theta_smoothing_scales):
    for i in range(kappa_maps.Nmaps):
        fname=kappa_maps.filenames[i]

        kappa_map = -kappa_maps.tomobins[i]
        map_mask_buffer[all_sky_map] = kappa_map[all_sky_map]
        if nside != 4096:
            map_out = hp.pixelfunc.ud_grade(map_mask_buffer, nside_out=4096, order_in='NESTED',dtype=np.float32)

            smoothed_map = smoothing(map_out,s)

        else:
            smoothed_map = smoothing(map_mask_buffer,s)

        wname = 'smoothed_theta%d_'%(s_i+1)+fname.split('/')[-1]
    
        np.save(dir_smoothed_maps+wname,np.array(smoothed_map))            