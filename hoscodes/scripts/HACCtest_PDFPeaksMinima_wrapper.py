import numpy as np
import os,sys
from HOScodes import *
from glob import glob
#cd ./HOS-Y1-prep/hoscodes/
#SkySim5000 maps

rseed = ['091','106']
zshells = ['101','26','34','51']
method ='lightcone_insitu'
#method = sys.argv[1] #[lightcone_insitu,rotflip]

method_output = {'lightcone_insitu':'LC_insitu','rotflip':'rotflip'}

mos = method_output[method]

method_output = {'lightcone_insitu':'LC_insitu','rotflip':'rotflip'}

mos = method_output[method]

print('method: %s'%mos)
for s in rseed:
    print('rseed: %s'%s)
    for c in zshells:
        print('Nshells: %s'%c)
        maps_names = '/pscratch/sd/j/jmena/HOS_Sims_test/HACC{0}_1024/shells_z{1}_{2}_nside8192/SRDv1_nz/'.format(s,c,method)
        
        dir_results = '/pscratch/sd/j/jatorres/data/HOScodes/HACC/HACC{0}_1024/shells_z{1}/{2}/'.format(s,c,mos)
        
        nside= 8192#
        #
        
        filenames = [maps_names + 'kappa_hacc_seed12345_nside8192_imap%d.fits'%i for i in range(3)]
        
        #filenames = np.sort(glob(fm))
        
        kappa_maps = kappacodes(dir_results=dir_results,
                                filenames=filenames,
                                nside=nside)
        
        #kappa_maps.readmaps_npy()
        kappa_maps.readmaps_fits()
    
        for i,map_i in enumerate(kappa_maps.mapbins):
            print('Tomobin: %d'%i)
        
            kappa_maps.run_PDFPeaksMinima(map_i,i)
            print('PDF,peaks,minima done...')
        
    #for j in range(i):
        #kappa_maps.run_PDFPeaksMinima(map_i,i,kappa_maps.mapbins[j],j,is_cross=True)
        #print('map2 cross %d %d done...'%(i,j))
