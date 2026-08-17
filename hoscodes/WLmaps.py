import numpy as np
import healpy as hp
import os,sys
from hoscodes.map_utils import *


class kappaMaps():
    """
    Class for reading maps (kappa)
    parameters:
        nside: Nside of map.
        filenames: NERSC location.
        smoothing_scale: If any. None by default (raw map)
    """
    def __init__(self,nside,filenames=None,mask=None,smoothing_scale=None):
        self.mask = mask
        self.nside= nside
        self.filenames = filenames
        self.smoothing_scale = smoothing_scale
        self.pixel_resolution = hp.pixelfunc.nside2resol(self.nside, arcmin=True)

    def loadMap(self,fn,file_format):
        """If reading only 1 map (no tomographic)"""
        if file_format == 'fits':
            self.kappamap = fits_readmap(fn)

        elif file_format == 'numpy':
            self.kappamap = numpy_readmap(fn)
         
        elif file_format == 'healpy':
            self.kappamap = healpy_readmap(fn)

        else:
            raise TypeError("only implemented fits, healpy and numpy types")        
        
    def MakeTomographic(self,file_format):
        """ Loads all tomographic maps from list: filenames"""
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
        

    

        

