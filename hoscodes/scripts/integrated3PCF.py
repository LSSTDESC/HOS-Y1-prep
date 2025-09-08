import numpy as np
import treecorr
import os,sys
import healpy as hp
from hoscodes.map_utils import
#import time
#start = time.time()

if __name__ == "__main__":

    filepath_map_data = sys.argv[1]
    p = sys.argv[2]
    is_shear_IA = False#bool(sys.argv[3])
    Ntomo1 = int(sys.argv[3])
    Ntomo2 = int(sys.argv[4])
    Ntomo3 = int(sys.argv[5])

    n_g='0.6'
    NSIDE=2048
    
    tomo_Map = 'tomo%d'%Ntomo1 # will be used for computing the aperture mass
    tomo_xiA = 'tomo%d'%Ntomo2 # will be used for computing position dependent 2PCF
    tomo_xiB = 'tomo%d'%Ntomo3 # will be used for computing position dependent 2PCF
    if is_shear_IA:
        g1_tomoMap_name = filepath_map_data+'g1+IA_GalMap_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_Map,n_g,NSIDE)
        g2_tomoMap_name = filepath_map_data+'g2+IA_GalMap_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_Map,n_g,NSIDE)
        g1_tomoxiA_name = filepath_map_data+'g1+IA_GalMap_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_xiA,n_g,NSIDE)
        g2_tomoxiA_name = filepath_map_data+'g2+IA_GalMap_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_xiA,n_g,NSIDE)
        g1_tomoxiB_name = filepath_map_data+'g1+IA_GalMap_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_xiB,n_g,NSIDE)
        g2_tomoxiB_name = filepath_map_data+'g2+IA_GalMap_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_xiB,n_g,NSIDE)
    else:
        g1_tomoMap_name = filepath_map_data+'g1_GalMap_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_Map,n_g,NSIDE)
        g2_tomoMap_name = filepath_map_data+'g2_GalMap_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_Map,n_g,NSIDE)
        g1_tomoxiA_name = filepath_map_data+'g1_GalMap_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_xiA,n_g,NSIDE)
        g2_tomoxiA_name = filepath_map_data+'g2_GalMap_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_xiA,n_g,NSIDE)
        g1_tomoxiB_name = filepath_map_data+'g1_GalMap_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_xiB,n_g,NSIDE)
        g2_tomoxiB_name = filepath_map_data+'g2_GalMap_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_xiB,n_g,NSIDE)
        
    w1_name = filepath_map_data+'weight_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_Map,n_g,NSIDE)
    w2_name = filepath_map_data+'weight_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_xiA,n_g,NSIDE)
    w3_name = filepath_map_data+'weight_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_xiB,n_g,NSIDE)
    
    footprint1_name = filepath_map_data+'footprint_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_Map,n_g,NSIDE)
    footprint2_name = filepath_map_data+'footprint_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_xiA,n_g,NSIDE)
    footprint3_name = filepath_map_data+'footprint_%s_%s_All_%sGpAM_nside_%d.fits'%(p,tomo_xiB,n_g,NSIDE)

    g1_Map, g2_Map, w_Map, footprint_Map = read_maps(g1_tomoMap_name,g2_tomoMap_name,w1_name,footprint1_name,p)
    g1_xiA, g2_xiA, w_xiA, footprint_xiA = read_maps(g1_tomoxiA_name,g2_tomoxiA_name,w2_name,footprint2_name,p)
    g1_xiB, g2_xiB, w_xiB, footprint_xiB = read_maps(g1_tomoxiB_name,g2_tomoxiB_name,w3_name,footprint3_name,p)

    do_2PCF = True
    min_sep_tc_xi = 0.5 # treecorr min sep in arcmins 
    max_sep_tc_xi = 120 # treecorr max sep in arcmins
    nbins_tc_xi = 20 # treecorr no. of bins (should be an integer)

    theta_Q_arcmins = 90 # patch radius in arcmins 
    theta_T_arcmins = 90 # patch radius in arcmins 
    min_sep_tc = 0.5 # treecorr min sep in arcmins 
    max_sep_tc = 120#2*theta_T_arcmins-10 # treecorr max sep in arcmins
    nbins_tc = 20 # treecorr no. of bins (should be an integer)

    # masking fraction for patch selection
    f_mask_max_Q = 0.2
    f_mask_max_W = 0.2

    pixel_size_arcmins = np.sqrt(hp.nside2pixarea(NSIDE, degrees=True))*60

    # set the angular bins in which to measure the local 2PCFs
    kk = treecorr.KKCorrelation(min_sep=min_sep_tc, max_sep=max_sep_tc, nbins=nbins_tc, sep_units='arcmin')
    theta = kk.rnom

    # settings for patch sizes
    
    theta_Q = np.radians( theta_Q_arcmins / 60 )
    theta_T = np.radians( theta_T_arcmins / 60 )
    #
    print('running: {0} {1}'.format(p,tomo_Map))
run_integrated3PCF(tomo_map=tomo_Map,tomo_xiA=tomo_xiA,tomo_xiB=tomo_xiB,theta_Q_arcmins=theta_Q_arcmins,theta_T_arcmins=theta_T_arcmins,NSIDE=NSIDE,n_g=n_g,filepath_map_data=filepath_map_data,p=p,is_IA=is_shear_IA)