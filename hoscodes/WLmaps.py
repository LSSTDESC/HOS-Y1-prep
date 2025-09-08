import numpy as np
import healpy as hp
import os,sys
from map_utils import *


class kappaMaps():
    """
    Class for reading maps (kappa)
    parameters:
        nside: Nside of map.
        filenames: NERSC location.
        smoothing_scale: If any. None by default (raw map)
    """
    def __init__(self,filenames,nside,smoothing_scale=None):
        self.nside= nside
        self.filenames = filenames
        self.smoothing_scale = smoothing_scale
        self.pixel_resolution = hp.pixelfunc.nside2resol(self.nside, arcmin=True)

    def MakeTomographic(self,file_format):
        if file_format == 'fits':
            self.tomobins = [fits_readmap(fn) for fn in self.filenames]

        elif file_format == 'numpy':
            self.tomobins = [numpy_readmap(fn) for fn in self.filenames]
         
        elif file_format == 'healpy':
            self.tomobins = [healpy_readmap(fn) for fn in self.filenames]

        else:
            raise TypeError("only implemented fits, healpy and numpy types")


class gammaMaps():
    """
    Class for reading maps (gamma)
    parameters:
        nside: Nside of map.
        filenames: NERSC location.
        smoothing_scale: If any. None by default (raw map)
    """
    
    def init(self,filenames,nside,smoothing_scale):
        self.nside=nside
        self.filenames = filenames
        self.smoothing_scale = smoothing_scale    
        self.pixel_resolution = healpy.pixelfunc.nside2resol(self.nside, arcmin=False)
        self.g1 = None
        self.g2 = None
        

    

        

