import numpy as np
import os,sys
from HOScodes import *
from glob import glob
#cd ./HOS-Y1-prep/hoscodes/
#SkySim5000 maps

IA_params_dict = {'noIA':['noIA'],
                  'NLA':['AIAp1','AIAm1'],
                  'deltaNLA':['AIAp1_bta1','AIAp1_bta2'],
                  'deltaTT':['C2m1_bta1','C2p1_bta1'],
                  'TATT':['AIAp1_C2p1_bta1'],
                  'TT':['C2p1','C2m1'],
                  'HODNLA':['AIAp1'],
                  'HODTT':['A2p1']}# If new model or free pars., please add here

maps_path = sys.argv[1]#'/global/cfs/cdirs/desc-wl/projects/wl-massmap/IA-infusion/SkySim5000/kappa/nz_SRD_KS/NLA/'
IA_model = sys.argv[2]#maps_path.split('/')[-2]

nside= 4096# We down the original resolution (NSIDE=8192) to the one used by the other maps
thetas=[4,8,16,32] #Mass aperture radii Map3
#
dir_results='/pscratch/sd/j/jatorres/data/HOScodes/SkySim5000IA/'+IA_model+'/'

free_par = IA_params_dict[IA_model]

for f in free_par:
    print('for params {0}'.format(f))
    if IA_model == 'noIA':
        fm = maps_path+'kappa_skysim5000_'+f+'_noisefree_*.npy'

    elif IA_model == 'HODNLA' or IA_model == 'HODTT':
        fm = maps_path+'kappa_skysim5000_'+f+'_'+IA_model+'_noisefree_*.npy'



    else:
        fm = maps_path+'kappa_skysim5000_'+IA_model+'_'+f+'_noisefree_*.npy'
    filenames = np.sort(glob(fm))

#filenames = os.listdir(raw_maps_path)
#map_names = [raw_maps_path+s for s in filenames]
      
    kappa_maps = kappacodes(dir_results=dir_results,
                            filenames=filenames,
                            nside=nside)
    #kappa_maps.readmaps_healpy()
    kappa_maps.readmaps_npy()#maps can be in npy,fits or healpy_fits(masked) format
       
    for i,map_i in enumerate(kappa_maps.mapbins):
        print('tomo bin %d'%i)
        #kappa_maps.run_map2alm(i)
        #print('map2 done...')
        kappa_maps.run_map3(i,thetas=thetas)
        print('map3 done...')
        
        print('Starting cross correlation...')
        
        for j in range(i):
            #kappa_maps.run_map2alm(Nmap1=i,Nmap2=j,is_cross=True)
            print('map2 cross %d %d done...'%(i,j))
        
        
