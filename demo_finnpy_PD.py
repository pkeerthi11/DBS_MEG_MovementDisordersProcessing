#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:46:10 2024

@author: prerana
"""
import os
import time
import mne
import scipy
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

fs_anatomy_path = '/storage/prerana/subjects/pd/fsaverage/'
fs_path = '/usr/local/freesurfer/7.4.1-1/'
subj_name = 'al0001a'
data_path = '/storage/prerana/subjects/pd/al0001a/01_tsss_DBS_GA_OFF1/01_tsss_DBS_GA_OFF1.fif'
anatomy_path = '/storage/prerana/subjects/pd/'
subj_anatomy_path = os.path.join(anatomy_path, subj_name) + '/'
sensor_cov_path = '/storage/prerana/sensor_covariance_file.fif'
cov_path = '/storage/prerana/sensor_covariance_file_processed'


finnpy_sr_utils.init_fs_paths(anatomy_path, fs_path)

overwrite_sensor_cov = False
visualize_coregistration = False
overwrite_ws_extract = False
visualize_skull_skin_plots = False
overwrite_mri_trans = False

fiducial_output = os.path.join(anatomy_path, subj_name, 'bem')
fiducial_output_no_bem = os.path.join(anatomy_path, subj_name)

if not os.path.exists(fiducial_output):
    os.mkdir(fiducial_output)

if not os.path.exists(os.path.join(fiducial_output, subj_name+'-'+'fiducials.fif')):
    _create_fiducials(fs_anatomy_path, fiducial_output_no_bem+"/", subj_name)

t1 = time.perf_counter()

(sensor_cov_eigen_val,
 sensor_cov_eigen_vec,
 sensor_cov_names) = finnpy_sr_sc.get_sensor_covariance(file_path = sensor_cov_path, 
                                                        cov_path = cov_path,
                                                        overwrite = overwrite_sensor_cov)
                                                        
rec_meta_info = mne.io.read_info(data_path)

(coreg_rotors,
 meg_pts) = finnpy_sr_coreg.calc_coreg(subj_name,subj_anatomy_path,
                                       rec_meta_info,
                                       registration_scale_type = "free")
                                       
finnpy_sr_mri_anat.scale_anatomy(subj_anatomy_path,
                                 subj_name, coreg_rotors[6:9])

(coreg_rotors,
 meg_pts) = finnpy_sr_coreg.calc_coreg(subj_name, subj_anatomy_path,
                                       rec_meta_info,
                                       registration_scale_type = "restricted")

rigid_mri_to_meg_trans = finnpy_sr_coreg.get_rigid_transform(coreg_rotors)
rigid_meg_to_mri_trans = scipy.linalg.inv(rigid_mri_to_meg_trans)
                                       
if (visualize_coregistration):
    finnpy_sr_coreg.plot_coregistration(rigid_mri_to_meg_trans, rec_meta_info,
                                      meg_pts, subj_anatomy_path)

finnpy_sr_bem.calc_skull_and_skin_models(subj_anatomy_path, subj_name,
                                         overwrite = overwrite_ws_extract)     

(ws_in_skull_vert,
 ws_in_skull_faces,
 ws_out_skull_vert,
 ws_out_skull_faces,
 ws_out_skin_vect,
 ws_out_skin_faces) = finnpy_sr_bem.read_skull_and_skin_models(subj_anatomy_path,
                                                               subj_name)


if (visualize_skull_skin_plots):
    finnpy_sr_bem.plot_skull_and_skin_models(ws_in_skull_vert,
                                             ws_in_skull_faces,
                                             ws_out_skull_vert,
                                             ws_out_skull_faces,
                                             ws_out_skin_vect,
                                             ws_out_skin_faces,
                                             subj_anatomy_path)
    
del ws_out_skull_vert; del ws_out_skull_faces
del ws_out_skin_vect; del ws_out_skin_faces

(in_skull_reduced_vert, in_skull_faces,
 in_skull_faces_area, in_skull_faces_normal,
 bem_solution) = finnpy_sr_bem.calc_bem_model_linear_basis(ws_in_skull_vert,
                                                           ws_in_skull_faces)
                                                           
                                                           
(lh_white_vert, lh_white_faces,
 rh_white_vert, rh_white_faces,
 lh_sphere_vert,
 rh_sphere_vert) = finnpy_sr_utils.read_cortical_models(subj_anatomy_path)                                                    

(octa_model_vert, octa_model_faces) = finnpy_sr_smm.create_source_mesh_model()

(lh_white_valid_vert,
 rh_white_valid_vert) = finnpy_sr_smm.match_source_mesh_model(lh_sphere_vert,
                                                              rh_sphere_vert,
                                                              octa_model_vert)
                                                              
                                                              
(fwd_sol,
 lh_white_valid_vert,
 rh_white_valid_vert) = finnpy_sr_fwd.calc_forward_model(lh_white_vert,
                                                         rh_white_vert,
                                                         rigid_meg_to_mri_trans,
                                                         rigid_mri_to_meg_trans,
                                                         rec_meta_info,
                                                         in_skull_reduced_vert,
                                                         in_skull_faces,
                                                         in_skull_faces_normal,
                                                         in_skull_faces_area,
                                                         bem_solution,
                                                         lh_white_valid_vert,
                                                         rh_white_valid_vert)    
                                                         
                                                         
optimized_fwd_sol = finnpy_sr_fwd.optimize_fwd_model(lh_white_vert, lh_white_faces,
                                                     lh_white_valid_vert, rh_white_vert,
                                                     rh_white_faces, rh_white_valid_vert,
                                                     fwd_sol, rigid_mri_to_meg_trans)                                                         


(inv_trans, noise_norm) = finnpy_sr_inv.calc_inverse_model(sensor_cov_eigen_val, sensor_cov_eigen_vec,
                                                sensor_cov_names, optimized_fwd_sol,
                                                rec_meta_info)

(fs_avg_trans_mat,
src_fs_avg_valid_lh_vert,
src_fs_avg_valid_rh_vert) = finnpy_sr_utils.get_mri_subj_to_fs_avg_trans_mat(lh_white_valid_vert,
                                                                             rh_white_valid_vert,
                                                                             octa_model_vert,
                                                                             subj_anatomy_path, fs_anatomy_path,
                                                                             overwrite = overwrite_mri_trans)


raw_data = mne.io.read_raw_fif(data_path, preload=True)    

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
    
    return data

ch_names = raw_data.info["ch_names"]
ch_types = __get_ch_types(raw_data.info)
fs = raw_data.info["sfreq"]
                                                                         
sen_data = raw_data.get_data()
sen_data = __remove_bad_channels(sen_data, ch_names, ch_types)
                                                                             
                                                                             
src_data = finnpy_sr_inv.apply_inverse_model(sen_data,
                                             inv_trans,
                                             noise_norm)

fs_avg_src_data = finnpy_sr_utils.apply_mri_subj_to_fs_avg_trans_mat(fs_avg_trans_mat,
                                                                     src_data)
(morphed_epoch_data,
 morphed_epoch_channels,
 morphed_region_names) = finnpy_sr_srm.apply_source_region_model (fs_avg_src_data,
                                                          src_fs_avg_valid_lh_vert,
                                                          src_fs_avg_valid_rh_vert,
                                                          octa_model_vert,
                                                          octa_model_faces,
                                                          fs_anatomy_path)
morphed_epoch_data = np.asarray(morphed_epoch_data)       


t2 = time.perf_counter()

time_diff = t2-t1
print(time_diff)                                                                      
                                                                             