import numpy as np
import os,sys
from HOScodes import *
from glob import glob
#cd ./HOS-Y1-prep/hoscodes/
#SkySim5000 maps

is_IA = False
if is_IA:

    IA_params_dict = {'noIA':['noIA'],
                      'NLA':['AIAp1','AIAm1'],
                      'deltaNLA':['noIA_bta1','AIAp1_bta1','AIAp1_bta2'],
                      'deltaTT':['C2m1_bta1','C2p1_bta1'],
                      'TATT':['AIAp1_C2p1_bta1'],
                      'TT':['C2p1','C2m1'],
                      'HODNLA':['AIAp1'],
                      'HODTT':['A2p1']}# If new model or free pars., please add here

maps_path = sys.argv[1]

nside= 4096# We down the original resolution (NSIDE=8192) to the one used by the other maps
thetas_MassAperture=[4,8,16,32] #Mass aperture radii Map3
#
dir_results='/pscratch/sd/j/jatorres/data/HOScodes/SkySim5000IA/deltaNLA/'

all_sky_map = np.load('/global/homes/j/jatorres/misc/mask_allsky.npy')

void_val = -1.6375000e+30
void_map = np.full(nside**2*12,void_val)

fm = maps_path + 'kappa_skysim5000_deltaNLA_noIA_bta1_noisefree_tomo*.dat.npy'
filenames = np.sort(glob(fm))

      
kappa_maps = kappacodes(dir_results=dir_results,
                        filenames=filenames,
                        nside=nside)
kappa_maps.readmaps_npy()#maps can be in npy,fits or healpy_fits(masked) format

for i,map_i in enumerate(kappa_maps.mapbins):
    print('tomo bin %d'%i)
    kappa_maps.run_map2alm(i)
    #print('map2 done...')
    kappa_maps.run_map3(i,thetas=thetas_MassAperture)
    print('map2, map3 done tomobin%d...'%i)

    #print('Starting cross correlation...')

    for j in range(i):
        kappa_maps.run_map2alm(Nmap1=i,Nmap2=j,is_cross=True)
        print('map2 cross %d %d done...'%(i,j))

