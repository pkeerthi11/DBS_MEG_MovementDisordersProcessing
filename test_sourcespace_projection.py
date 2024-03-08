#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 09:30:48 2024

@author: prerana
"""

import os

import mne
import time
import math
import scipy
from scipy.sparse import save_npz, load_npz
import numpy as np
import finnpy.source_reconstruction.utils as finnpy_sr_utils
import finnpy.source_reconstruction.mri_anatomy as finnpy_sr_mri_anat
import finnpy.source_reconstruction.sensor_covariance as finnpy_sr_sc
from finnpy.source_reconstruction.mri_anatomy import _create_fiducials
import finnpy.source_reconstruction.coregistration_meg_mri as finnpy_sr_coreg
import finnpy.source_reconstruction.bem_model as finnpy_sr_bem
import finnpy.source_reconstruction.source_mesh_model as finnpy_sr_smm
import finnpy.source_reconstruction.forward_model as finnpy_sr_fwd
import finnpy.source_reconstruction.inverse_model as finnpy_sr_inv
import finnpy.source_reconstruction.source_region_model as finnpy_sr_srm 

import finnpy.misc.timed_pool as tp

def __get_ch_types(info):
    ch_types = list()
    for ch_idx in range(len(info["chs"])):
        if (int(info["chs"][ch_idx]["kind"]) == 910):
            ch_types.append("ias")
        elif (int(info["chs"][ch_idx]["kind"]) == 900):
            ch_types.append("syst")
        elif (int(info["chs"][ch_idx]["kind"]) == 2):
            ch_types.append("eeg")
        elif (int(info["chs"][ch_idx]["kind"]) == 1):
            if (int(info["chs"][ch_idx]["coil_type"]) == 3012):
                ch_types.append("grad")
            elif (int(info["chs"][ch_idx]["coil_type"]) == 3024):
                ch_types.append("mag")
        else:
            ch_types.append("misc")
        
    return ch_types

def __remove_bad_channels(data, ch_names, ch_types):
    bad_idxs = list()
    for (ch_name_idx, ch_name) in enumerate(ch_names):
        if (ch_name[:3] != "MEG"):
            bad_idxs.append(ch_name_idx)
    for bad_idx in bad_idxs[::-1]:
        ch_names.pop(bad_idx)
        ch_types.pop(bad_idx)
    data = np.delete(data, bad_idxs, axis = 0)
    
    return data, ch_names

def transform_data_matrix(data_matrix_split, inv_trans, noise_norm, fs_avg_trans_mat, 
                          src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert,
                          octa_model_vert, octa_model_faces):
    
    print('apply inv model')
    src_data = finnpy_sr_inv.apply_inverse_model(data_matrix_split,
                                                 inv_trans,
                                                 noise_norm)
    
    print('apply mri subj to fs avg trans mat')
    fs_avg_src_data = finnpy_sr_utils.apply_mri_subj_to_fs_avg_trans_mat(fs_avg_trans_mat,
                                                                         src_data)
    del data_matrix_split
    del src_data 
    print('apply source region model')
    (morphed_epoch_data,
     morphed_epoch_channels,
     morphed_region_names) = finnpy_sr_srm.apply_source_region_model (fs_avg_src_data,
                                                              src_fs_avg_valid_lh_vert,
                                                              src_fs_avg_valid_rh_vert,
                                                              octa_model_vert,
                                                              octa_model_faces,
                                                              fs_anatomy_path)
    
    return morphed_epoch_data, morphed_epoch_channels, morphed_region_names

fs_anatomy_path = '/usr/local/freesurfer/7.4.1-1/subjects/fsaverage/'
anatomy_path = '/storage/prerana/subjects/dy/'
fs_path =  '/usr/local/freesurfer/7.4.1-1/'
data_path = '/storage/prerana/subjects/dy/al0059a/01_tsss_1_REST1/01_tsss_1_REST1.fif'
subj_name = 'al0059a'
intermediate_save_path = os.path.join(os.path.split(data_path)[0], 'sourceRecIntermediateFiles')



inv_trans_path = os.path.join(intermediate_save_path, 
                              subj_name+'-'+'inv_trans.npy')
noise_norm_path = os.path.join(intermediate_save_path,
                                subj_name+'-'+'noise_norm.npy')
fs_avg_trans_mat_path = os.path.join(intermediate_save_path,
                                subj_name+'-'+'fs_avg_trans_mat.npz')
src_fs_avg_valid_lh_vert_path = os.path.join(intermediate_save_path,
                                subj_name+'-'+'src_fs_avg_valid_lh_vert.npy')
src_fs_avg_valid_rh_vert_path = os.path.join(intermediate_save_path,
                                subj_name+'-'+'src_fs_avg_valid_rh_vert.npy')
octa_model_vert_path = os.path.join(intermediate_save_path,
                                subj_name+'-'+'octa_model_vert.npy')
octa_model_faces_path = os.path.join(intermediate_save_path,
                                subj_name+'-'+'octa_model_faces.npy')


finnpy_sr_utils.init_fs_paths(anatomy_path, fs_path)

raw_data = mne.io.read_raw_fif(data_path, preload=True) 
if raw_data.times[-1] > 200:
    long_recording = True
else:
    long_recording = False
    #return (None, None, None)

ch_names = raw_data.info["ch_names"]
ch_types = __get_ch_types(raw_data.info)
                                                                         
sen_data = raw_data.get_data()
sen_data, ch_names = __remove_bad_channels(sen_data, ch_names, ch_types)

inv_trans = np.load(inv_trans_path)
noise_norm = np.load(noise_norm_path)
fs_avg_trans_mat = load_npz(fs_avg_trans_mat_path)
src_fs_avg_valid_lh_vert = np.load(src_fs_avg_valid_lh_vert_path)
src_fs_avg_valid_rh_vert = np.load(src_fs_avg_valid_rh_vert_path)
octa_model_vert = np.load(octa_model_vert_path)
octa_model_faces = np.load(octa_model_faces_path)

sen_data = sen_data[:,:10000]

sen_data[ch_names.index("MEG1513"), :] = -2


(morphed_epoch_data, 
 morphed_epoch_channels, 
 morphed_region_names) = transform_data_matrix(sen_data, inv_trans, noise_norm, fs_avg_trans_mat, 
                           src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert,
                           octa_model_vert, octa_model_faces) 


np.save('test_transformed_data_1513.npy', morphed_epoch_data)                                               
                                               
import finnpy.visualization.topoplot as tp
import matplotlib
matplotlib.use("Qtagg")
import matplotlib.pyplot as plt
topo = tp.Topoplot("MEG") 
topo.run(values = sen_data, ch_name_list = ch_names, annotate_ch_names = True, substitute_channels = [], omit_channels = [])
plt.show()