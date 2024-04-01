import numpy as np
import healpy as hp
import healsparse
from healpy.sphtfunc import anafast
from healpy.pixelfunc import ud_grade
import os,sys

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
        
#    def readmaps_fits(self,nside_c=32):
#        self.nside_c = nside_c
#        a = [healsparse.HealSparseMap.read(l,nside_coverage=32) for l in self.filenames]
#        self.mapbins=[a_.generate_healpix_map() for a_ in a]
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
        
        if is_cross:
            map1_smooth = np.append(map1_smooth,map2_smooth,axis=0)

        bins=np.linspace(-0.1-0.001,0.1+0.001,201) 
        binmids=(bins[1:]+bins[:-1])/2

        #create histograms
        
        counts_smooth,bins=np.histogram(map1_smooth,density=True,bins=bins)
        #counts,bins=np.histogram(kappa_data,density=True,bins=bins)
        
        #find the peak positions and amplitudes
        map1_smooth_ma = hp.ma(map1_smooth)
        peak_pos, peak_amp = find_extrema(map1_smooth_ma,lonlat=True)
        #repeat for minima
        minima_pos, minima_amp = find_extrema(map1_smooth_ma,minima=True,lonlat=True)
        
        peaks = np.vstack([peak_pos.T,peak_amp]).T
        minima = np.vstack([minima_pos.T,minima_amp]).T
        
        fn_header = self.filenames[Nmap1].split('/')[-1]
        fn_out_counts = self.dir_results+'PDF/'+fn_header.split('.')[0]+'_Nmap%d_Counts_kappa_width0.1_200Kappabins.dat'%(Nmap1+1)
        fn_out_minima = self.dir_results+'peaks/'+fn_header.split('.')[0]+'_Nmap%d_minima_posRADEC_amp.dat'%(Nmap1+1)
        fn_out_peaks = self.dir_results+'minima/'+fn_header.split('.')[0]+'_Nmap%d_peaks_posRADEC_amp.dat'%(Nmap1+1)
        
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
    
    def run_integrated3PCF(self,tomo_map,tomo_xiA,tomo_xiB,theta_Q_arcmins,theta_T_arcmins,NSIDE,n_g,extension):
        
        g1_Map, g2_Map, w_Map, footprint_Map = read_maps(tomo_Map)
        g1_xiA, g2_xiA, w_xiA, footprint_xiA = read_maps(tomo_xiA)
        g1_xiB, g2_xiB, w_xiB, footprint_xiB = read_maps(tomo_xiB)
        
        filepath_output = './measurements/shear_Q'+str(int(theta_Q_arcmins))+'W'+str(int(theta_T_arcmins))+'W'+str(int(theta_T_arcmins))+'/'+tomo_Map+'_'+tomo_xiA+'_'+tomo_xiB+'_All_'+n_g+'GpAM/NSIDE_'+str(NSIDE)+'/position_dependent_outputs'+extension+'/'

        if (os.path.isdir(filepath_output) == False):
            os.makedirs(filepath_output)
        
        pixel_size_arcmins = np.sqrt(hp.nside2pixarea(NSIDE, degrees=True))*60

        if ('bin_slop_0' in extension):
            bin_slop_val = 0
            extension = '_' + extension
        elif ('default_bin_slop' in extension):
            extension = '_' + extension
        
        kk = treecorr.KKCorrelation(min_sep=min_sep_tc, max_sep=max_sep_tc,
                                    nbins=nbins_tc, sep_units='arcmin')
        theta = kk.rnom

        # settings for patch sizes
        theta_Q = np.radians( theta_Q_arcmins / 60 )
        theta_T = np.radians( theta_T_arcmins / 60 )
        
        footprint_HR = footprint_Map
        patch_radius_Q = theta_Q
        patch_radius_W = theta_T

        NSIDE_HR = hp.npix2nside(footprint_HR.size)

        NSIDE_LR = 32
        all_pixels_indices = np.arange(0,hp.nside2npix(NSIDE_LR),1)
        footprint_LR = hp.ud_grade(footprint_HR, NSIDE_LR)
        footprint_mask_LR = (footprint_LR != 0)

        footprint_pixels_angle = hp.pix2ang(NSIDE_LR, all_pixels_indices[footprint_mask_LR])
        footprint_pixels_theta = footprint_pixels_angle[0]
        footprint_pixels_phi = footprint_pixels_angle[1]

        mask_footprint_pixels_indices = np.zeros(footprint_pixels_theta.size).astype('bool')

        print('Number of patch centers being considered from low resolution mask =', footprint_pixels_theta.size)
        
        for i in range(footprint_pixels_theta.size):

            #print('\nPatch center #'+str(i+1))

            center_footprint_HR = hp.ang2vec(footprint_pixels_theta[i], footprint_pixels_phi[i])
            all_pixels_indices_footprint_HR_Q = hp.query_disc(NSIDE_HR, center_footprint_HR, 5.1*patch_radius_Q)
            all_pixels_indices_footprint_HR_W = hp.query_disc(NSIDE_HR, center_footprint_HR, patch_radius_W)

            # tomo bin 1
            pixel_values_footprint_HR_Q = footprint_HR[all_pixels_indices_footprint_HR_Q]
            mask_footprint_HR_Q = (pixel_values_footprint_HR_Q == 0)
            masked_pixels_indices_footprint_HR_Q = all_pixels_indices_footprint_HR_Q[mask_footprint_HR_Q]
            f_mask_Q = masked_pixels_indices_footprint_HR_Q.size/all_pixels_indices_footprint_HR_Q.size

            pixel_values_footprint_HR_W = footprint_HR[all_pixels_indices_footprint_HR_W]
            mask_footprint_HR_W = (pixel_values_footprint_HR_W == 0)
            masked_pixels_indices_footprint_HR_W = all_pixels_indices_footprint_HR_W[mask_footprint_HR_W]
            f_mask_W = masked_pixels_indices_footprint_HR_W.size/all_pixels_indices_footprint_HR_W.size

            #print('f_mask (Q, W): '+str(f_mask_Q)+' '+str(f_mask_W))

            # only accept those patch centers for which the patch has masking less than f_mask_max
            if ((f_mask_Q <= f_mask_max_Q and f_mask_W <= f_mask_max_W)):
                mask_footprint_pixels_indices[i] = True
            else:
                mask_footprint_pixels_indices[i] = False

        accepted_footprint_pixels_theta = footprint_pixels_theta[mask_footprint_pixels_indices]
        accepted_footprint_pixels_phi = footprint_pixels_phi[mask_footprint_pixels_indices]

        patch_count = accepted_footprint_pixels_theta.size

        print('Number of patch centers accepted =', patch_count)

        patch_centers_theta = accepted_footprint_pixels_theta
        patch_centers_phi = accepted_footprint_pixels_phi

        # Real parts
        all_patches_M_a_Re = np.zeros([patch_count])
        all_patches_xi_pp_Re = np.zeros([patch_count, nbins_tc]) 
        all_patches_xi_mm_Re = np.zeros([patch_count, nbins_tc]) 

        # Imaginary parts
        all_patches_M_a_Im = np.zeros([patch_count])
        all_patches_xi_pp_Im = np.zeros([patch_count, nbins_tc])
        all_patches_xi_mm_Im = np.zeros([patch_count, nbins_tc]) 

        for i in range(patch_count):
        #for i in range(10):
            print("Computing aperture mass and position-dependent 2PCF at location of patch # ", i+1)

            pixel_index_patch = hp.ang2pix(NSIDE, patch_centers_theta[i], patch_centers_phi[i])

            patch_center = hp.ang2vec(patch_centers_theta[i], patch_centers_phi[i])
            pixels_indices_patch = hp.query_disc(NSIDE, patch_center, theta_T)

            ##################################################
            ## shear position-dependent aperture mass
            Q_patch_center_dec = np.array([np.pi/2 - hp.pix2ang(NSIDE, pixel_index_patch)[0]])
            Q_patch_center_RA = np.array([hp.pix2ang(NSIDE, pixel_index_patch)[1]])
            pixels_indices_Q_patch = hp.query_disc(NSIDE, patch_center, 5*theta_Q)

            mask_Q_patch = (footprint_Map[pixels_indices_Q_patch] != 0) 
            pixels_indices_Q_patch = pixels_indices_Q_patch[mask_Q_patch]

            total_pixels_Q_patch = pixels_indices_Q_patch.size

            pixels_dec_Q_patch = np.pi/2 - hp.pix2ang(NSIDE, pixels_indices_Q_patch)[0]
            pixels_RA_Q_patch = hp.pix2ang(NSIDE, pixels_indices_Q_patch)[1]

            cat_Q_patch_center = treecorr.Catalog(ra=Q_patch_center_RA, dec=Q_patch_center_dec, ra_units='rad', dec_units='rad')

            cat_Q_patch_shear = treecorr.Catalog(ra=pixels_RA_Q_patch, dec=pixels_dec_Q_patch, ra_units='rad', dec_units='rad', 
                                                 g1=g1_Map[pixels_indices_Q_patch], g2=g2_Map[pixels_indices_Q_patch], 
                                                 w=w_Map[pixels_indices_Q_patch], flip_g1=True)

            if ('bin_slop_0' in extension):
                NG = treecorr.NGCorrelation(min_sep=pixel_size_arcmins, max_sep=5*theta_Q_arcmins, nbins=100,
                                            bin_type='Linear', sep_units='arcmins', bin_slop=bin_slop_val)
            elif ('default_bin_slop' in extension):
                NG = treecorr.NGCorrelation(min_sep=pixel_size_arcmins, max_sep=5*theta_Q_arcmins, nbins=100,
                                            bin_type='Linear', sep_units='arcmins')

            NG.process(cat_Q_patch_center, cat_Q_patch_shear) 

            theta_bins_arr = np.radians( NG.rnom / 60 )
            delta_theta = np.radians( (NG.right_edges-NG.left_edges) / 60 )

            Q_T_arr = np.zeros(theta_bins_arr.size)
            for j in range(theta_bins_arr.size):
                Q_T_arr[j] = Q_T(theta_bins_arr[j], theta_Q)

            all_patches_M_a_Re[i] = np.sum(2*np.pi*delta_theta*theta_bins_arr*NG.xi*Q_T_arr)
            all_patches_M_a_Im[i] = np.sum(2*np.pi*delta_theta*theta_bins_arr*NG.xi_im*Q_T_arr)

            ##################################################
            ## shear position-dependent 2PCF

            mask_patch = (footprint_xiA[pixels_indices_patch] != 0) & (footprint_xiB[pixels_indices_patch] != 0)
            pixels_indices_patch = pixels_indices_patch[mask_patch]

            total_pixels_patch = pixels_indices_patch.size

            pixels_dec_patch = np.pi/2 - hp.pix2ang(NSIDE, pixels_indices_patch)[0]
            pixels_RA_patch = hp.pix2ang(NSIDE, pixels_indices_patch)[1]

            cat_xiA = treecorr.Catalog(ra=pixels_RA_patch, dec=pixels_dec_patch, ra_units='rad', dec_units='rad',
                                          g1=g1_xiA[pixels_indices_patch], g2=g2_xiA[pixels_indices_patch], w=w_xiA[pixels_indices_patch],
                                          flip_g1=True)
            cat_xiB = treecorr.Catalog(ra=pixels_RA_patch, dec=pixels_dec_patch, ra_units='rad', dec_units='rad',
                                          g1=g1_xiB[pixels_indices_patch], g2=g2_xiB[pixels_indices_patch], w=w_xiB[pixels_indices_patch],
                                          flip_g1=True)

            if ('bin_slop_0' in extension):
                GG = treecorr.GGCorrelation(min_sep=min_sep_tc, max_sep=max_sep_tc, nbins=nbins_tc, sep_units='arcmin', bin_slop=bin_slop_val)
            elif ('default_bin_slop' in extension):
                GG = treecorr.GGCorrelation(min_sep=min_sep_tc, max_sep=max_sep_tc, nbins=nbins_tc, sep_units='arcmin')
            GG.process(cat_xiA, cat_xiB) 

            all_patches_xi_pp_Re[i] = GG.xip
            all_patches_xi_mm_Re[i] = GG.xim

            all_patches_xi_pp_Im[i] = GG.xip_im
            all_patches_xi_mm_Im[i] = GG.xim_im
        

        np.save(filepath_output+'/all_patches_M_a_Re.npy', all_patches_M_a_Re)
        np.save(filepath_output+'/all_patches_xi_pp_Re.npy', all_patches_xi_pp_Re)
        np.save(filepath_output+'/all_patches_xi_mm_Re.npy', all_patches_xi_mm_Re)

        # mean

        M_a_Re = np.mean(all_patches_M_a_Re)

        xi_pp_Re = np.mean(all_patches_xi_pp_Re, axis=0)
        xi_mm_Re = np.mean(all_patches_xi_mm_Re, axis=0)

        zeta_app_Re = np.mean(np.vstack(all_patches_M_a_Re)*all_patches_xi_pp_Re, axis=0) - M_a_Re*xi_pp_Re  
        zeta_amm_Re = np.mean(np.vstack(all_patches_M_a_Re)*all_patches_xi_mm_Re, axis=0) - M_a_Re*xi_mm_Re

        header = 'theta, M_a_Re, xi_pp_Re, xi_mm_Re, zeta_app_Re, zeta_amm_Re'
        dat = np.array([theta, 
                        np.ones(theta.size)*M_a_Re,
                        xi_pp_Re, xi_mm_Re, 
                        zeta_app_Re, zeta_amm_Re])
        dat = dat.T

        np.savetxt(filepath_output+'/average_over_patches_Re_footprint.dat', dat, delimiter = ' ', header=header)

        # stddev
        M_a_Re_std = np.std(all_patches_M_a_Re)

        xi_pp_Re_std = np.std(all_patches_xi_pp_Re, axis=0)
        xi_mm_Re_std = np.std(all_patches_xi_mm_Re, axis=0)

        zeta_app_Re_std = np.std(np.vstack(all_patches_M_a_Re)*all_patches_xi_pp_Re, axis=0)  
        zeta_amm_Re_std = np.std(np.vstack(all_patches_M_a_Re)*all_patches_xi_mm_Re, axis=0)

        header = 'theta, M_a_Re_std, xi_pp_Re_std, xi_mm_Re_std, zeta_app_Re_std, zeta_amm_Re_std'
        dat = np.array([theta, 
                        np.ones(theta.size)*M_a_Re_std,
                        xi_pp_Re_std, xi_mm_Re_std, 
                        zeta_app_Re_std, zeta_amm_Re_std]) / np.sqrt(patch_count)
        dat = dat.T

        np.savetxt(filepath_output+'/stddev_of_mean_over_patches_Re_footprint.dat', dat, delimiter = ' ', header=header)

        # covariance

        xi_pp_Re_cov = np.cov(all_patches_xi_pp_Re.T)
        xi_mm_Re_cov = np.cov(all_patches_xi_mm_Re.T)

        zeta_app_Re_cov = np.cov((np.vstack(all_patches_M_a_Re)*all_patches_xi_pp_Re).T)  
        zeta_amm_Re_cov = np.cov((np.vstack(all_patches_M_a_Re)*all_patches_xi_mm_Re).T)  

        np.savetxt(filepath_output+'/cov_over_patches_xi_pp_Re_footprint.dat', xi_pp_Re_cov, delimiter = ' ')
        np.savetxt(filepath_output+'/cov_over_patches_xi_mm_Re_footprint.dat', xi_mm_Re_cov, delimiter = ' ')

        np.savetxt(filepath_output+'/cov_over_patches_zeta_app_Re_footprint.dat', zeta_app_Re_cov, delimiter = ' ')
        np.savetxt(filepath_output+'/cov_over_patches_zeta_amm_Re_footprint.dat', zeta_amm_Re_cov, delimiter = ' ')

        ### Imaginary parts

        np.save(filepath_output+'/all_patches_M_a_Im.npy', all_patches_M_a_Im)
        np.save(filepath_output+'/all_patches_xi_pp_Im.npy', all_patches_xi_pp_Im)
        np.save(filepath_output+'/all_patches_xi_mm_Im.npy', all_patches_xi_mm_Im)

        # mean

        M_a_Im = np.mean(all_patches_M_a_Im)

        xi_pp_Im = np.mean(all_patches_xi_pp_Im, axis=0)
        xi_mm_Im = np.mean(all_patches_xi_mm_Im, axis=0)

        zeta_app_Im = np.mean(np.vstack(all_patches_M_a_Im)*all_patches_xi_pp_Im, axis=0) - M_a_Im*xi_pp_Im  
        zeta_amm_Im = np.mean(np.vstack(all_patches_M_a_Im)*all_patches_xi_mm_Im, axis=0) - M_a_Im*xi_mm_Im

        header = 'theta, M_a_Im_std, xi_pp_Im_std, xi_mm_Im_std, zeta_app_Im_std, zeta_amm_Im_std'
        dat = np.array([theta, 
                        np.ones(theta.size)*M_a_Im,
                        xi_pp_Im, xi_mm_Im, 
                        zeta_app_Im, zeta_amm_Im])
        dat = dat.T

        np.savetxt(filepath_output+'/average_over_patches_Im_footprint.dat', dat, delimiter = ' ', header=header)

        # stddev

        M_a_Im_std = np.std(all_patches_M_a_Im)

        xi_pp_Im_std = np.std(all_patches_xi_pp_Im, axis=0)
        xi_mm_Im_std = np.std(all_patches_xi_mm_Im, axis=0)

        zeta_app_Im_std = np.std(np.vstack(all_patches_M_a_Im)*all_patches_xi_pp_Im, axis=0)  
        zeta_amm_Im_std = np.std(np.vstack(all_patches_M_a_Im)*all_patches_xi_mm_Im, axis=0)

        header = 'theta, M_a_Im, xi_pp_Im, xi_mm_Im, zeta_app_Im, zeta_amm_Im'
        dat = np.array([theta, 
                        np.ones(theta.size)*M_a_Im_std,
                        xi_pp_Im_std, xi_mm_Im_std, 
                        zeta_app_Im_std, zeta_amm_Im_std]) / np.sqrt(patch_count)
        dat = dat.T

        np.savetxt(filepath_output+'/stddev_of_mean_over_patches_Im_footprint.dat', dat, delimiter = ' ', header=header)

        # covariance

        xi_pp_Im_cov = np.cov(all_patches_xi_pp_Im.T)
        xi_mm_Im_cov = np.cov(all_patches_xi_mm_Im.T)

        zeta_app_Im_cov = np.cov((np.vstack(all_patches_M_a_Im)*all_patches_xi_pp_Im).T)  
        zeta_amm_Im_cov = np.cov((np.vstack(all_patches_M_a_Im)*all_patches_xi_mm_Im).T)

        np.savetxt(filepath_output+'/cov_over_patches_xi_pp_Im_footprint.dat', xi_pp_Im_cov, delimiter = ' ')
        np.savetxt(filepath_output+'/cov_over_patches_xi_mm_Im_footprint.dat', xi_mm_Im_cov, delimiter = ' ')

        np.savetxt(filepath_output+'/cov_over_patches_zeta_app_Im_footprint.dat', zeta_app_Im_cov, delimiter = ' ')
        np.savetxt(filepath_output+'/cov_over_patches_zeta_amm_Im_footprint.dat', zeta_amm_Im_cov, delimiter = ' ')

        ### angular separations

        np.savetxt(filepath_output+'/angular_separations_arcmin.dat', theta.T)
       #### TODO: move all routines (see notebooks) to independent python file


        