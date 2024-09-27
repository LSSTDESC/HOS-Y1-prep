import numpy as np
import os,sys
from HOScodes import *
from glob import glob
#cd ./HOS-Y1-prep/hoscodes/
#SkySim5000 maps

maps_names = sys.argv[1]
IA_model = sys.argv[2]#maps_names.split('/')[-2] 

is_IA = True
if is_IA:

    IA_params_dict = {'noIA':['noIA'],
                      'NLA':['AIAp1','AIAm1'],
                      'deltaNLA':['noIA_bta1','AIAp1_bta1','AIAp1_bta2'],
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

nside= 4096#
#
theta_smoothing_scale_bins = [1,2,3,4,5]


dir_results='/pscratch/sd/j/jatorres/data/HOScodes/SkySim5000IA/'+IA_model+'/'

for _,s_l in enumerate(theta_smoothing_scale_bins):

    filenames = [maps_names + 'smoothed_theta%d_kappa_skysim5000_'%(s_l)+prefix+'_noisefree_tomo%d.dat.npy'%i for i in range(1,6)]
    
    #filenames = np.sort(glob(fm))

    kappa_maps = kappacodes(dir_results=dir_results,
                            filenames=filenames,
                            nside=nside)

    kappa_maps.readmaps_npy()

    for i,map_i in enumerate(kappa_maps.mapbins):
        print('Tomobin: %d'%i)

        #kappa_maps.run_PDFPeaksMinima(map_i,i)
        print('PDF,peaks,minima done...')
        
        for j in range(i):
            kappa_maps.run_PDFPeaksMinima(map_i,i,kappa_maps.mapbins[j],j,is_cross=True)
            print('map2 cross %d %d done...'%(i,j))
