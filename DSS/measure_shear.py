import numpy as np
from astropy.table import Table
import healpy as hp
import treecorr
from tqdm import tqdm

nside = 2**12


load_prepared_data = False
if(load_prepared_data):
    pix=np.load('data/pix_nside'+str(nside)+'.npy')
    gamma1=np.load('data/gamma1_nside'+str(nside)+'.npy')[pix]
    gamma2=np.load('data/gamma2_nside'+str(nside)+'.npy')[pix]

else:
    fn_map_curved = '/global/homes/j/jharno/IA-infusion/SkySim5000/kappa/1.0060kappa.npy'
    hpmap = np.load(fn_map_curved)
    nside_low=2**12
    hpmap_low=hp.pixelfunc.ud_grade(hpmap,nside_out=nside,power=-2)
    pix=np.where(hpmap_low!=-0)

    fn_gamma1="/global/homes/j/jharno/IA-infusion/SkySim5000/shear/1.0060gamma1.npy"
    fn_gamma1 = np.load(fn_gamma1)
    nside_low=2**12
    gamma1=hp.pixelfunc.ud_grade(fn_gamma1,nside_out=nside)[pix]
    
    fn_gamma2="/global/homes/j/jharno/IA-infusion/SkySim5000/shear/1.0060gamma2.npy"
    fn_gamma2 = np.load(fn_gamma2)
    gamma2=hp.pixelfunc.ud_grade(fn_gamma2,nside_out=nside)[pix]
    

ra,dec=hp.pix2ang(nside=nside,ipix=pix,lonlat=True)
Nap=np.load('data/Nap_tophat20_nside'+str(nside)+'.npy')
Nap_table = Table(np.array([ra[0],dec[0],Nap[pix][0]]).T, names=('ra','dec','Nap'))
Nap_table.sort('Nap')

source_catalog = treecorr.Catalog(g1=-gamma1[0], g2=gamma2[0],ra=ra[0], dec=dec[0],ra_units='deg', dec_units='deg')
ng = treecorr.NGCorrelation(min_sep=5, max_sep=120, nbins=20, sep_units='arcmin',bin_slop=0.02,bin_type="Log",metric='Arc')

N_quantiles = 5
len_quantile = len(pix[0])//N_quantiles
for quantile in tqdm(range(N_quantiles)):
    ra_gal = Nap_table['ra'][len_quantile*quantile:len_quantile*(quantile+1)]
    dec_gal = Nap_table['dec'][len_quantile*quantile:len_quantile*(quantile+1)]
    lens_catalog = treecorr.Catalog(ra=ra_gal, dec=dec_gal,ra_units='deg', dec_units='deg')

    ng.process(lens_catalog,source_catalog)
    ng.write('data/shear/shear_quant'+str(quantile+1)+'_tophat_nside'+str(nside)+'.dat')
