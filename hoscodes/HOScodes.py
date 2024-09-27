import numpy as np
import healpy as hp
import healsparse
from healpy.sphtfunc import anafast
from healpy.pixelfunc import ud_grade
import os,sys
import treecorr

sys.path.append('/global/homes/j/jatorres/HOS-Y1-prep/Map3/')#replace with __init__.py
sys.path.append('/global/homes/j/jatorres/HOS-Y1-prep/PDF_Peaks_Minima/')
sys.path.append('/global/homes/j/jatorres/HOS-Y1-prep/DSS_clean/')
from aperture_mass_computer import measureMap3FromKappa
from Peaks_minima import find_extrema
from DSS_functions import DSS_class

#nside_c = 32

class kappamaps():
    """
    Class for reading maps (kappa) and store the relevant information
    parameters:
        nside: Nside of map.
    """
    def __init__(self,filenames,nside):
        self.nside=nside
        self.filenames = filenames
        self.mapbins = []
        
    def readmaps_fits(self,nside_c=32):
        self.nside_c = nside_c
        a = [healsparse.HealSparseMap.read(l,nside_coverage=32) for l in self.filenames]
        self.mapbins=[a_.generate_healpix_map(nside=self.nside,nest=False) for a_ in a]
    def readmaps_npy(self):
        self.mapbins = [np.load(l) for l in self.filenames]        
    def readmaps_healpy(self):
        self.mapbins = [hp.read_map(l) for l in self.filenames]
        
    def smoothing(self,mapbin,sl,is_map=True):
        """
        map smoothing in healpy applied to self.mapbins in class.
        -----------
        parameters:
        sl: smoothing lenght in arcmins.
        
        """
        
        if is_map:
            kappa_map = mapbin
        else:
            kappa_map = self.mapbins[mapbin]
        sl_rad = sl/60/180*np.pi
        kappa_masked = hp.ma(kappa_map)
        return hp.smoothing(kappa_masked,sigma = sl_rad) #smooth map with a Gaussian filter with std = sl_rad 
#        return smoothed_mapbins
        #save the smoothed map so we don't have to do this everytime we start the notebook

class kappacodes(kappamaps):
    def __init__(self,dir_results,filenames,nside):
        super().__init__(filenames,nside)
        self.Nmaps = len(self.filenames)
        self.dir_results = dir_results#os.path.join(os.getcwd())
        if not os.path.exists(self.dir_results):
            os.makedirs(self.dir_results)

    def run_map2alm(self,Nmap1,Nmap2=None,is_cross=False):
        if is_cross:
            map1 = self.mapbins[Nmap1]
            map2 = self.mapbins[Nmap2]
            Cl = anafast(map1=map1, map2=map2, nspec=None, 
             lmax=5000, mmax=None, iter=1,
             alm=False,pol=False, use_weights=False,
             datapath=None, gal_cut=0,use_pixel_weights=False)
            Cl *= 8.0
            fn_header = self.filenames[Nmap1].split('/')[-1]
            fn_out = self.dir_results+'map2/'+fn_header.split('.')[0]+'_Nmap%d_%d_map2_Cell_ell_0_5000.dat'%(Nmap1+1,Nmap2+1)
            np.savetxt(fn_out,Cl)
        
        
        else:
            map1 = self.mapbins[Nmap1]
            Cl = anafast(map1, map2=None, nspec=None, 
                         lmax=5000, mmax=None, iter=1,
                         alm=False,pol=False, use_weights=False,
                         datapath=None, gal_cut=0,use_pixel_weights=False)
            Cl *= 8.0
            fn_header = self.filenames[Nmap1].split('/')[-1]
            fn_out = self.dir_results+'map2/'+fn_header.split('.')[0]+'_Nmap%d_map2_Cell_ell_0_5000.dat'%(Nmap1+1)
            np.savetxt(fn_out,Cl)
        
        return Cl

    def run_map3(self,Nmap1,thetas,is_cross=False):
        fn_header = self.filenames[Nmap1].split('/')[-1]
        fn_out = self.dir_results+'map3/'+fn_header.split('.')[0] + '_map3_DV_thetas.dat'
        measureMap3FromKappa(self.mapbins[Nmap1], thetas=thetas, nside=self.nside, fn_out=fn_out, verbose=False, doPlots=False)
        results_map3 = np.loadtxt(fn_out)
        #self.map3.append(results_map3)
        return results_map3
    
    def run_PDFPeaksMinima(self,map1_smooth,Nmap1,map2_smooth=None,Nmap2=None,is_cross=False):
        bins=np.linspace(-0.1-0.001,0.1+0.001,201) 
        binmids=(bins[1:]+bins[:-1])/2

        if is_cross:
            map1_smooth_ma = hp.ma(map1_smooth)
            map2_smooth_ma = hp.ma(map2_smooth)
            map1_smooth_ma = map1_smooth_ma+map2_smooth_ma
            
            counts_smooth,bins=np.histogram(map1_smooth_ma,density=True,bins=bins)
            peak_pos, peak_amp = find_extrema(map1_smooth_ma,lonlat=True)
            #repeat for minima
            minima_pos, minima_amp = find_extrema(map1_smooth_ma,minima=True,lonlat=True)

            peaks = np.vstack([peak_pos.T,peak_amp]).T
            minima = np.vstack([minima_pos.T,minima_amp]).T            
            fn_header = self.filenames[Nmap1].split('/')[-1]
            fn_out_counts = self.dir_results+'PDF/'+fn_header.split('.')[0]+'_Nmap%d_Nmap%d_Counts_kappa_width0.1_200Kappabins.dat'%(Nmap1+1,Nmap2+1)
            fn_out_minima = self.dir_results+'peaks/'+fn_header.split('.')[0]+'_Nmap%d_Nmap%d_minima_posRADEC_amp.dat'%(Nmap1+1,Nmap2+1)
            fn_out_peaks = self.dir_results+'peaks/'+fn_header.split('.')[0]+'_Nmap%d_Nmap%d_peaks_posRADEC_amp.dat'%(Nmap1+1,Nmap2+1)

        else:
            #create histograms
            map1_smooth_ma = hp.ma(map1_smooth)

            counts_smooth,bins=np.histogram(map1_smooth_ma,density=True,bins=bins)

            #find the peak positions and amplitudes
            peak_pos, peak_amp = find_extrema(map1_smooth_ma,lonlat=True)
            #repeat for minima
            minima_pos, minima_amp = find_extrema(map1_smooth_ma,minima=True,lonlat=True)

            peaks = np.vstack([peak_pos.T,peak_amp]).T
            minima = np.vstack([minima_pos.T,minima_amp]).T

            fn_header = self.filenames[Nmap1].split('/')[-1]
            fn_out_counts = self.dir_results+'PDF/'+fn_header.split('.')[0]+'_Nmap%d_Counts_kappa_width0.1_200Kappabins.dat'%(Nmap1+1)
            fn_out_minima = self.dir_results+'peaks/'+fn_header.split('.')[0]+'_Nmap%d_minima_posRADEC_amp.dat'%(Nmap1+1)
            fn_out_peaks = self.dir_results+'peaks/'+fn_header.split('.')[0]+'_Nmap%d_peaks_posRADEC_amp.dat'%(Nmap1+1)
        
        np.savetxt(fn_out_counts,counts_smooth)
        np.savetxt(fn_out_minima,minima)
        np.savetxt(fn_out_peaks,peaks)
        
        return counts_smooth,peaks,minima

    
class gammaMaps():
    def init(self,filenames,nshells,seed,nzs,nside):
        self.nside=nside
        self.filenames = filenames
        self.Nbins = len(self.filenames)
        
    def run_DSS(density_contrast_map,shear_table,pix,gamma1,gamm2,load_prepared_data = False):

        DSS_class = DSS_fct.DSS_class(filter='top_hat',theta_ap=20,nside=nside)
        Nap=DSS_class.calc_Nap(bias=1.5,n0=0.3,density_contrast=density_contrast)
        ra,dec=hp.pix2ang(nside=nside,ipix=pix,lonlat=True)
        Nap_table = Table(np.array([ra[0],dec[0],Nap[pix][0]]).T, names=('ra','dec','Nap'))
        gamma_table = Table(np.array([ra[0],dec[0],-gamma1[0],gamma2[0]]).T, names=('ra','dec','gamma1','gamma2'))
        shear_table=DSS_class.calc_shear(Nap_table=Nap_table,gamma_table=gamma_table,theta_min=5,theta_max=120,nbins=20,N_quantiles=5)
        
        return Nap_table,shear_table
    
