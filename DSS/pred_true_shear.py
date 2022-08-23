from __future__ import print_function
import numpy as np
import ctypes
import sys
from astropy.table import Table
import multiprocessing
import time

path_2_DSS_code = 'DSS_code_bias'

sys.path.append(path_2_DSS_code)
import DSS_modul

filter_version='tophat'
print(filter_version)
cosmology_dict = {'Omega_m': 0.2648,'Omega_b': 0.0448, 'sigma_8': 0.801,'h_100': 0.71, 'w_0': -1.0, 'w_a': 0.0, 'n_s': 0.963}
systematics_dict = {'bias_1': 1.5, 'alpha0_1':1 ,'bias_cor_1': 0, 'S3_enhance': 0.0, 'dz_lens': 0.0, 'dz_s1': 0.0, 'dz_s2': 0.0, 'dz_s3': 0.0, 'b_s1': 0.0, 'b_s2': 0.0, 'b_s3': 0.0 }
output_dict = {'theta_min': 0.4, 'theta_max': 150, 'n_bin': 40, 'N_quantile': 5}
#modus P_m(k) T(K)
#0 internal halofit Eisenstein&Hu
#1 pyccl halofit Eisenstein&Hu
#2 pyccl halofit bbks
#3 pyccl halofit boltzmann_camb
#4 pyccl MiraTitan Eisenstein&Hu

if(filter_version=='tophat'):
    catalogue_dict = {'n0_1': 0.4082301197036551, 'files_to_lens_bin1': 'nofz/n_of_z_lens.dat', 'n_source_bins': 1, 'files_to_source_bin1': 'nofz/n_of_z_source.dat', 'files_to_source_bin2': '', 'files_to_source_bin3': ''}
    filter_dict = {'filter_size': 20, 'file_to_filter': 'filters/tophat_20.dat', 'filter_bins': 1, 'Nap_norm_min': 0, 'Nap_norm_max': 4000,'PDF_modus': 1 }
    powerspectrum_dict = {'modus': 1, 'k_cut_in_h_over_Mpc': 0.0, 'identity_1': 'outputs_4_powerspectrum/OuterRim','delta_mU_min':-1.5,'delta_mU_max':3}

if(filter_version=='adapted'):
    catalogue_dict = {'n0_1': 0.4082301197036551, 'files_to_lens_bin1': 'nofz/n_of_z_lens.dat', 'n_source_bins': 1, 'files_to_source_bin1': 'nofz/n_of_z_source.dat', 'files_to_source_bin2': '', 'files_to_source_bin3': ''}
    filter_dict = {'filter_size': 120, 'file_to_filter': 'filters/adapted_filter.dat', 'filter_bins': 60, 'Nap_norm_min': -750, 'Nap_norm_max': 1250,'PDF_modus': 1  }
    powerspectrum_dict = {'modus': 1, 'k_cut_in_h_over_Mpc': 0.0, 'identity_1': 'outputs_4_powerspectrum/OuterRim','delta_mU_min':-0.015,'delta_mU_max':0.03}


DSS_modul = DSS_modul.DSS_modul(cosmology_dict=cosmology_dict,systematics_dict=systematics_dict,catalogue_dict=catalogue_dict,output_dict=output_dict,filter_dict=filter_dict,powerspectrum_dict=powerspectrum_dict, path_2_DSS_code=path_2_DSS_code)

start = time.time()
print(start)

theta_values=DSS_modul.get_theta_values()
shear_table = Table()
shear_table.add_column(theta_values,name=r'r_arcmin')

Nap_values=DSS_modul.get_Nap_values()
P_Nap_table = Table()
P_Nap_table.add_column(Nap_values,name=r'Nap_values')
number_of_Nap_values = len(Nap_values)

results = DSS_modul.predict_model(lbin=1)
for quant in range(output_dict['N_quantile']):
    shear_table.add_column(results[output_dict['n_bin']*quant:output_dict['n_bin']*(quant+1)],name=r'g_t,'+str(quant+1)+'_sbin'+str(4))

P_Nap_table.add_column(results[output_dict['n_bin']* output_dict['N_quantile'] : output_dict['n_bin']* output_dict['N_quantile'] +number_of_Nap_values],name=r'pofNap')

path = 'data/pred/shear_'+filter_version+'_best.fits'
shear_table.write(path, format='fits',overwrite=True)

path = 'data/pred/pofNap_'+filter_version+'_best.fits'
P_Nap_table.write(path, format='fits',overwrite=True)

print(time.time()-start)
