import numpy as np
import healpy as hp
from glob import glob
import multiprocessing

nside = 4096
nzs='Euclid_dndz_fu08_bin1-5.dat'
file_names = np.sort(glob('/global/homes/j/jharno/IA-infusion/SkySim5000/kappa/V0/kappa_tomo*.npy'))[:-1]

def fixing_maps(file_name):
    name = file_name.split('/')[-1]
    map_in = np.load(file_name)
    for i in range(len(map_in)):
        if map_in[i] == '-0.0':
            map_in[i] = -1.6375e+30
    map_out = hp.pixelfunc.ud_grade(map_in=map_in,
                              nside_out = nside,
                              pess=False,
                              order_in='RING',
                              order_out=None,
                              power=None,
                              dtype=np.float32)
    hp.write_map("/global/cscratch1/sd/jatorres/KappaMapsLR/SkySim5000/SkySim5000_"+nzs+"_"+name.split('.')[0]+".fits",map_out)
    return -9

print("masking and fixing resolution...")
pool = multiprocessing.Pool(processes=5)
Fn_r = np.zeros((len(file_names),1))
Fn_r[:] = pool.map(fixing_maps,file_names)
print("done... data saved in /global/cscratch1/sd/jatorres/KappaMapsLR/SkySim5000/")    
