import numpy as np
import os,sys
from glob import glob
from treecorr import Catalog,GGCorrelation
from healsparse import HealSparseMap

class readshearsmaps():
    def __init__(self,dir_name,config):
        self.config = config
        self.filenames = [os.path.join(dir_name, file) for file in os.listdir(dir_name)]
        self.tomobins = [Catalog(l, config, flip_g2=True)
                         for l in self.filenames]
        self.Ntomobins = len(self.filenames)
        
class readkappamaps():
    def __init__(self,dir_name,nshells,seeds,nzs):
        self.nshells = nshells
        self.seeds = seeds
        self.nzs = nzs
        self.filenames = [os.path.join(path, f'shells_z{nshells}_subsampleauto_groupiso/{nzs}/kappa_hacc_nz{tomo}_nside4096_seed{seed}.fits') for tomo in range(1,6)]
        self.tomobins = [healsparse.HealSparseMap.read(l) for l in self.filenames]

        
        
        