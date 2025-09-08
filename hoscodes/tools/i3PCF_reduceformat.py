import numpy as np
import sys 

IA_params_dict = {'noIA':['noIA'],
                  'NLA':['AIAp1','AIAm1'],
                  'deltaNLA':['AIAp1_bta1','AIAp1_bta2','noIA_bta1'],
                  'deltaTT':['C2p1_bta1','C2m1_bta1'],
                  'TATT':['AIAp1_C2p1_bta1'],
                  'TT':['C2p1','C2m1'],
                  'HODNLA':['AIAp1'],
                  'HODTT':['A2p1']}

is_IA = False


IA_models = ['NLA','TT','deltaNLA','deltaTT','HODNLA','HODTT']
#BCM_models = ['dmo','dmb','dbm_Mc2.5e13','b1']
BCM_models = ['dmb_theta_ej_2', 
              'dmb_theta_ej_3', 
              'dmb_theta_ej_5', 
              'dmb_theta_ej_6']#dmo for reference

root = '/pscratch/sd/j/jatorres/data/GalCat/BCM/'
root_out = '/pscratch/sd/j/jatorres/data/HOScodes/BCM/'

models = BCM_models

trios = [[1,1,1],
         [3,3,3],
         [2,2,2],
         [4,4,4],
         [5,5,5]]
]# iterate i j k

#trios = [[1,1,1],
#         [1,1,3],
#         [1,3,3],
#         [3,1,1],
#         [3,1,3],
#         [3,3,3],
#         [2,2,2],
#         [4,4,4],
#         [5,5,5]
]# iterate i j k

for model in models:

    i = sys.argv[1]
    j = sys.argv[2]
    k = sys.argv[3]


    tomo_Map = 'tomo%d'%i # will be used for computing the aperture mass
    tomo_xiA = 'tomo%d'%j # will be used for computing position dependent 2PCF
    tomo_xiB = 'tomo%d'%k # will be used for computing position dependent 2PCF

    f = 0

    if is_IA:

        if model == 'noIA':
            prefix = model           

        else:
            p = IA_params_dict[model][f]
            prefix = model + '_' + p

        prefix += '_shear+IA'
        filepath_input = root+model+'/shear+IA_'+prefix+'_'+tomo_Map+'_'+tomo_xiA+'_'+tomo_xiB+'_All_0.6/'             

    else:
        prefix = model
        filepath_input = root+model+'/shear_'+prefix+'_'+tomo_Map+'_'+tomo_xiA+'_'+tomo_xiB+'_All_0.6/'


    out_name = root_out+model+'/i3PCF/'

    dat_Re = np.loadtxt(filepath_input+prefix+'_average_over_patches_Re_footprint.dat')
    dat_Im = np.loadtxt(filepath_input+prefix+'_average_over_patches_Im_footprint.dat')

    dat_Re_std = np.loadtxt(filepath_input+prefix+'_stddev_of_mean_over_patches_Re_footprint.dat')
    dat_Im_std = np.loadtxt(filepath_input+prefix+'_stddev_of_mean_over_patches_Im_footprint.dat')

    N = 1052 # number of patches
    xi_pp_cov = np.loadtxt(filepath_input+prefix+'_cov_over_patches_xi_pp_Re_footprint.dat')/N
    xi_mm_cov = np.loadtxt(filepath_input+prefix+'_cov_over_patches_xi_mm_Re_footprint.dat')/N
    zeta_app_cov = np.loadtxt(filepath_input+prefix+'_cov_over_patches_zeta_app_Re_footprint.dat')/N
    zeta_amm_cov = np.loadtxt(filepath_input+prefix+'_cov_over_patches_zeta_amm_Re_footprint.dat')/N

    Xp = np.array([dat_Re[:,0],dat_Re[:,2],np.sqrt(np.diag(xi_pp_cov))]).T
    Xm = np.array([dat_Re[:,0],dat_Re[:,3],np.sqrt(np.diag(xi_mm_cov))]).T
    np.savetxt(out_name+'xi_p_SkySim5000_'+tomo_Map+'_'+tomo_xiA+'_'+tomo_xiB+'_'+prefix+'_theta_xi_xierr.dat',Xp)
    np.savetxt(out_name+'xi_m_SkySim5000_'+tomo_Map+'_'+tomo_xiA+'_'+tomo_xiB+'_'+prefix+'_theta_xi_xierr.dat',Xm)

    Zp = np.array([dat_Re[:,0],dat_Re[:,4],np.sqrt(np.diag(zeta_app_cov))]).T
    Zm = np.array([dat_Re[:,0],dat_Re[:,5],np.sqrt(np.diag(zeta_app_cov))]).T
    np.savetxt(out_name+'zeta_p_SkySim5000_'+tomo_Map+'_'+tomo_xiA+'_'+tomo_xiB+'_'+prefix+'_theta_zeta_zetaerr.dat',Zp)
    np.savetxt(out_name+'zeta_m_SkySim5000_'+tomo_Map+'_'+tomo_xiA+'_'+tomo_xiB+'_'+prefix+'_theta_zeta_zetaerr.dat',Zm)
   