import numpy as np
from astropy.table import Table
import healpy as hp
import treecorr
from tqdm import tqdm

nside=2**12

load_prepared_data = False
if(load_prepared_data):
    density_contrast = np.load('data/density_contrast.npy')
else:
    density_map = np.load('/global/homes/j/jharno/IA-infusion/SkySim5000/density/density_map_331_dens_allsky.npy')
    density_contrast_map = (density_map-density_map.mean())/density_map.mean()

n0 = 0.3
bias = 1.5
n = n0*(1+bias*density_contrast)

negative_n=np.where(n<0)[0]
n[negative_n]=0
SGD = np.random.poisson(n)

pixel=np.where(SGD>0)[0]
positive_SGD = SGD[pixel]

SGD=hp.pixelfunc.ud_grade(SGD,nside_out=nside,power=-2)

def top_hat(b, radius):
    return np.where(abs(b)<=radius, 1, 0)

filter_size=20

pixel_area_inrad = hp.pixelfunc.nside2pixarea(nside=nside)

b = np.linspace(0,np.radians(filter_size/60*1.1),10000)
bw = top_hat(b, np.radians(filter_size/60))
beam = hp.sphtfunc.beam2bl(bw, b, nside*3)
Nap = hp.smoothing(SGD, beam_window=beam)/pixel_area_inrad
np.save('data/Nap_tophat20_nside'+str(nside),Nap)