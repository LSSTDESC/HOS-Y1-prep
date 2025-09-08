#cd ~/HOS-Y1-prep/hoscodes/
import numpy as np
import healpy as hp
from glob import glob
from HOScodes import kappamaps,kappacodes
import os,sys

maps_names = sys.argv[1]
IA_model = sys.argv[2]#maps_names.split('/')[-2] 

is_IA = True
if is_IA:

    IA_params_dict = {'noIA':['noIA'],
                      'NLA':['AIAp1','AIAm1'],
                      'deltaNLA':['noAI_bta1','AIAp1_bta1','AIAp1_bta2'],
                      'deltaTT':['C2p1_bta1','C2m1_bta1'],
                      'TATT':['AIAp1_C2p1_bta1'],
                      'TT':['C2p1','C2m1'],
                      'HODNLA':['AIAp1'],
                      'HODTT':['A2p1']}
    f = int(sys.argv[3])

    if IA_model == 'noIA':
        prefix = IA_model

    else:
        param = IA_params_dict[IA_model][f]
        prefix = IA_model + '_' + param 


filenames = [maps_names+'kappa_skysim5000_'+prefix+'_noisy_tomo%d.dat.npy'%i for i in range(1,6)]

theta_smoothing_scales = [1,2,5,10.25,20] #define the smoothing length for the map in arcmins 
nside= 8192# We down the original resolution (NSIDE=8192) to the one used by the other maps

dir_smoothed_maps = '/pscratch/sd/j/jatorres/KappaMaps/Smoothed_Maps/SkySim5000IA/'+IA_model+'/'
dir_results = './'

all_sky_map = np.load('/global/homes/j/jatorres/misc/mask_allsky.npy')

void_val = -1.6375000e+30
void_map = np.full(nside**2*12,void_val)

kappa_maps = kappacodes(dir_results,filenames,nside)
kappa_maps.readmaps_npy()
for s_i,s in enumerate(theta_smoothing_scales):
    for i in range(kappa_maps.Nmaps):
        fname=kappa_maps.filenames[i]
        map_mask_buffer =  void_map.copy()
        kappa_map = kappa_maps.mapbins[i]
        map_mask_buffer[all_sky_map] = kappa_map[all_sky_map]
        map_out = hp.pixelfunc.ud_grade(map_mask_buffer, nside_out=4096, order_in='NESTED',dtype=np.float32)
        smoothed_map = kappa_maps.smoothing(map_out,s)
        wname = 'smoothed_theta%d_'%(s_i+1)+fname.split('/')[-1]
        
        np.save(dir_smoothed_maps+wname,np.array(smoothed_map))