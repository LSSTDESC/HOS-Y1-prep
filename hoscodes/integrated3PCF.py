import numpy as np
import treecorr
import os,sys
import healpy as hp
#import time
#start = time.time()

def read_maps(tomo_bin,p):
    
    g1_map = hp.read_map(filepath_map_data+'GalMap_'+p+'_'+tomo_bin+'_All_'+n_g+'GpAM_nside_'+str(NSIDE)+'_g1_IA.fits')
    g2_map = hp.read_map(filepath_map_data+'GalMap_'+p+'_'+tomo_bin+'_All_'+n_g+'GpAM_nside_'+str(NSIDE)+'_g2_IA.fits')
    w_map = hp.read_map(filepath_map_data+'GalMap_'+p+'_'+tomo_bin+'_All_'+n_g+'GpAM_nside_'+str(NSIDE)+'_w.fits')
    footprint_map = hp.read_map(filepath_map_data+'GalMap_'+p+'_'+tomo_bin+'_All_'+n_g+'GpAM_nside_'+str(NSIDE)+'_footprint.fits')
    
    return g1_map, g2_map, w_map, footprint_map

def pixel2RaDec(index, nside, nest=False):
    #converts healpy pixel index to ra dec (in degrees)
    theta, phi = hp.pixelfunc.pix2ang(nside ,index,nest=nest)
    return np.degrees(phi), -np.degrees(theta-np.pi/2.0)

def RaDec2pixel(ra, dec, nside):
    #converts ra dec to healpy pixel index 
    #assumes that input ra and dec are in radians
    return hp.pixelfunc.ang2pix(nside, np.pi/2.0 + np.radians(-dec), np.radians(ra))

def Q_T(theta, theta_Q):
    # eqn (3.2) of http://articles.adsabs.harvard.edu/pdf/2005IAUS..225...81K

    return theta**2/(4*np.pi*theta_Q**4)*np.exp(-theta**2/(2*theta_Q**2))

def run_integrated3PCF(tomo_map,tomo_xiA,tomo_xiB,theta_Q_arcmins,theta_T_arcmins,NSIDE,n_g,extension,filepath_map_data,p):

    g1_Map, g2_Map, w_Map, footprint_Map = read_maps(tomo_Map,p)
    g1_xiA, g2_xiA, w_xiA, footprint_xiA = read_maps(tomo_xiA,p)
    g1_xiB, g2_xiB, w_xiB, footprint_xiB = read_maps(tomo_xiB,p)

    filepath_output = filepath_map_data+'measurements/shear_Q'+str(int(theta_Q_arcmins))+'W'+str(int(theta_T_arcmins))+'W'+str(int(theta_T_arcmins))+'/'+tomo_Map+'_'+tomo_xiA+'_'+tomo_xiB+'_All_'+n_g+'GpAM/NSIDE_'+str(NSIDE)+'/position_dependent_outputs'+extension+'/'+p+'_'

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


    np.save(filepath_output+'weIA_all_patches_M_a_Re.npy', all_patches_M_a_Re)
    np.save(filepath_output+'weIA_all_patches_xi_pp_Re.npy', all_patches_xi_pp_Re)
    np.save(filepath_output+'weIA_all_patches_xi_mm_Re.npy', all_patches_xi_mm_Re)

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

    np.savetxt(filepath_output+'weIA_average_over_patches_Re_footprint.dat', dat, delimiter = ' ', header=header)

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

    np.savetxt(filepath_output+'weIA_stddev_of_mean_over_patches_Re_footprint.dat', dat, delimiter = ' ', header=header)

    # covariance

    xi_pp_Re_cov = np.cov(all_patches_xi_pp_Re.T)
    xi_mm_Re_cov = np.cov(all_patches_xi_mm_Re.T)

    zeta_app_Re_cov = np.cov((np.vstack(all_patches_M_a_Re)*all_patches_xi_pp_Re).T)  
    zeta_amm_Re_cov = np.cov((np.vstack(all_patches_M_a_Re)*all_patches_xi_mm_Re).T)  

    np.savetxt(filepath_output+'weIA_cov_over_patches_xi_pp_Re_footprint.dat', xi_pp_Re_cov, delimiter = ' ')
    np.savetxt(filepath_output+'weIA_cov_over_patches_xi_mm_Re_footprint.dat', xi_mm_Re_cov, delimiter = ' ')

    np.savetxt(filepath_output+'weIA_cov_over_patches_zeta_app_Re_footprint.dat', zeta_app_Re_cov, delimiter = ' ')
    np.savetxt(filepath_output+'weIA_cov_over_patches_zeta_amm_Re_footprint.dat', zeta_amm_Re_cov, delimiter = ' ')

    ### Imaginary parts

    np.save(filepath_output+'weIA_all_patches_M_a_Im.npy', all_patches_M_a_Im)
    np.save(filepath_output+'weIA_all_patches_xi_pp_Im.npy', all_patches_xi_pp_Im)
    np.save(filepath_output+'weIA_all_patches_xi_mm_Im.npy', all_patches_xi_mm_Im)

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

    np.savetxt(filepath_output+'weIA_average_over_patches_Im_footprint.dat', dat, delimiter = ' ', header=header)

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

    np.savetxt(filepath_output+'weIA_stddev_of_mean_over_patches_Im_footprint.dat', dat, delimiter = ' ', header=header)

    # covariance

    xi_pp_Im_cov = np.cov(all_patches_xi_pp_Im.T)
    xi_mm_Im_cov = np.cov(all_patches_xi_mm_Im.T)

    zeta_app_Im_cov = np.cov((np.vstack(all_patches_M_a_Im)*all_patches_xi_pp_Im).T)  
    zeta_amm_Im_cov = np.cov((np.vstack(all_patches_M_a_Im)*all_patches_xi_mm_Im).T)

    np.savetxt(filepath_output+'weIA_cov_over_patches_xi_pp_Im_footprint.dat', xi_pp_Im_cov, delimiter = ' ')
    np.savetxt(filepath_output+'weIA_cov_over_patches_xi_mm_Im_footprint.dat', xi_mm_Im_cov, delimiter = ' ')

    np.savetxt(filepath_output+'weIA_cov_over_patches_zeta_app_Im_footprint.dat', zeta_app_Im_cov, delimiter = ' ')
    np.savetxt(filepath_output+'weIA_cov_over_patches_zeta_amm_Im_footprint.dat', zeta_amm_Im_cov, delimiter = ' ')

    ### angular separations

    np.savetxt(filepath_output+'weIA_angular_separations_arcmin.dat', theta.T)
    
    return None

def calculate_patch_radius(patch_area_sq_degrees):
    return math.acos(1-patch_area_sq_degrees*np.pi/(2*180*180))

    
if __name__ == "__main__":
    
    filepath_map_data = sys.argv[1]
    p = sys.argv[2]
    n_g='0.6'
    NSIDE=2048
    Ntomo1 = int(sys.argv[3])
    Ntomo2 = int(sys.argv[4])
    Ntomo3 = int(sys.argv[5])
    
    tomo_Map = 'tomo%d'%Ntomo1 # will be used for computing the aperture mass
    tomo_xiA = 'tomo%d'%Ntomo2 # will be used for computing position dependent 2PCF
    tomo_xiB = 'tomo%d'%Ntomo3 # will be used for computing position dependent 2PCF

    g1_Map, g2_Map, w_Map, footprint_Map = read_maps(tomo_Map,p)
    g1_xiA, g2_xiA, w_xiA, footprint_xiA = read_maps(tomo_xiA,p)
    g1_xiB, g2_xiB, w_xiB, footprint_xiB = read_maps(tomo_xiB,p)

    extension = 'bin_slop_0' # e.g. bin_slop_0, default_bin_slop
    if ('bin_slop_0' in extension):
        bin_slop_val = 0
        extension = '_' + extension
    elif ('default_bin_slop' in extension):
        extension = '_' + extension

    do_2PCF = True
    min_sep_tc_xi = 5 # treecorr min sep in arcmins 
    max_sep_tc_xi = 170 # treecorr max sep in arcmins
    nbins_tc_xi = 15 # treecorr no. of bins (should be an integer)

    theta_Q_arcmins = 90 # patch radius in arcmins 
    theta_T_arcmins = 90 # patch radius in arcmins 
    min_sep_tc = 5 # treecorr min sep in arcmins 
    max_sep_tc = 2*theta_T_arcmins-10 # treecorr max sep in arcmins
    nbins_tc = 15 # treecorr no. of bins (should be an integer)

    # masking fraction for patch selection
    f_mask_max_Q = 0.2
    f_mask_max_W = 0.2


    filepath_output = filepath_map_data+'measurements/shear_Q'+str(int(theta_Q_arcmins))+'W'+str(int(theta_T_arcmins))+'W'+str(int(theta_T_arcmins))+'/'+tomo_Map+'_'+tomo_xiA+'_'+tomo_xiB+'_All_'+n_g+'GpAM/NSIDE_'+str(NSIDE)+'/position_dependent_outputs'+extension+'/'

    if (os.path.isdir(filepath_output) == False):
        os.makedirs(filepath_output)

    pixel_size_arcmins = np.sqrt(hp.nside2pixarea(NSIDE, degrees=True))*60
    #extension = '_bin_slop'
    if ('bin_slop_0' in extension):
        bin_slop_val = 0
        extension = '_' + extension
    elif ('default_bin_slop' in extension):
        extension = '_' + extension
    # set the angular bins in which to measure the local 2PCFs
    kk = treecorr.KKCorrelation(min_sep=min_sep_tc, max_sep=max_sep_tc, nbins=nbins_tc, sep_units='arcmin')
    theta = kk.rnom

    # settings for patch sizes
    
    theta_Q = np.radians( theta_Q_arcmins / 60 )
    theta_T = np.radians( theta_T_arcmins / 60 )
    print('running: {0} {1}'.format(p,tomo_Map))
run_integrated3PCF(tomo_map=tomo_Map,tomo_xiA=tomo_xiA,tomo_xiB=tomo_xiB,theta_Q_arcmins=theta_Q_arcmins,theta_T_arcmins=theta_T_arcmins,NSIDE=NSIDE,n_g=n_g,extension=extension,filepath_map_data=filepath_map_data,p=p)