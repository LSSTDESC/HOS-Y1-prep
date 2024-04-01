#cd ~/HOS-Y1-prep/hoscodes/
import numpy as np
import healpy as hp
from glob import glob
from HOScodes import kappamaps,kappacodes
import os,sys

is_IA = False
if is_IA:

    IA_params_dict = {'noIA':['noIA'],
                      'NLA':['AIAp1','AIAm1'],
                      'deltaNLA':['AIAp1_bta1','AIAp1_bta2'],
                      'deltaTT':['C2m1_bta1','C2p1_bta1'],
                      'TATT':['AIAp1_C2p1_bta1'],
                      'TT':['C2p1','C2m1'],
                      'HODNLA':['AIAp1'],
                      'HODTT':['A2p1']}

    #maps_path = sys.argv[1]#'/global/cfs/cdirs/desc-wl/projects/wl-massmap/IA-infusion/SkySim5000/kappa/nz_SRD_KS/NLA/'
    #IA_model = sys.argv[2]#maps_path.split('/')[-2]
    #free_par = IA_params_dict[IA_model]

maps_path = '/global/cfs/cdirs/desc-wl/projects/wl-massmap/IA-infusion/HACC-Y1/kappa-KS/nz_SRD_KS/'#sys.argv[1]#'/global/cfs/cdirs/desc-wl/projects/wl-massmap/IA-infusion/SkySim5000/kappa/nz_SRD_KS/NLA/'
    
sl_arcmin= [5,8,10] #define the smoothing length for the map in arcmins 
nside= 4096# We down the original resolution (NSIDE=8192) to the one used by the other maps

dir_smoorhed_maps = '/pscratch/sd/j/jatorres/KappaMaps/Smoothed_Maps/HACC-Y1/'
dir_results = '/pscratch/sd/j/jatorres/data/HOScodes/HACC-Y1/'

all_sky_map = np.load('/global/homes/j/jatorres/mask_allsky.npy')
#all_sky_map = np.load('/global/cfs/cdirs/desc-wl/projects/wl-massmap/IA-infusion/SkySim5000/kappa/nz_SRD_KS/HODTT/HOD_mask_4096.dat.npy').astype(bool)

void_val = -1.6375000e+30
void_map = np.full(nside**2*12,void_val)


#for f in free_par:
#    print('for params {0}'.format(f))
#    if IA_model == 'noIA':
#        fm = maps_path+'kappa_skysim5000_'+f+'_noisefree_*.npy'
#    elif IA_model == 'HODNLA' or IA_model == 'HODTT':
#        fm = maps_path+'kappa_skysim5000_'+f+'_'+IA_model+'_noisefree_*.npy'
        
#    else:
#        fm = maps_path+'kappa_skysim5000_'+IA_model+'_'+f+'_noisefree_*.npy'
fm = maps_path + 'kappa_nzshells*'
filenames = np.sort(glob(fm))

kappa_maps = kappacodes(dir_results,filenames,nside)
kappa_maps.readmaps_npy()
for s_i,s in enumerate(sl_arcmin):
    for i in range(kappa_maps.Nmaps):
        fname=kappa_maps.filenames[i]
        map_mask_buffer =  void_map.copy()
        kappa_map = -1*kappa_maps.mapbins[i]
        map_mask_buffer[all_sky_map] = kappa_map[all_sky_map]
        smoothed_map = kappa_maps.smoothing(map_mask_buffer,s)
        wname = 'smoothed_theta%d_'%(s_i+1)+fname.split('/')[-1]
        np.save(dir_smoorhed_maps+wname,np.array(smoothed_map))