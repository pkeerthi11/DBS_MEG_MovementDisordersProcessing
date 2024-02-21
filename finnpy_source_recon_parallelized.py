#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:16:49 2024

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
    
    return data


def subject_specific_processing(subj_name):
    overwrite_ws_extract = False
    anatomy_path = '/storage/prerana/subjects/VerifySourceReconstructionPipeline/'
    subj_anatomy_path = os.path.join(anatomy_path, subj_name) + '/'
    bem_save_path = os.path.join(subj_anatomy_path, 'bem')
    
    finnpy_sr_bem.calc_skull_and_skin_models(subj_anatomy_path, subj_name,
                                             overwrite = overwrite_ws_extract)     

    (ws_in_skull_vert,
      ws_in_skull_faces,
      ws_out_skull_vert,
      ws_out_skull_faces,
      ws_out_skin_vect,
      ws_out_skin_faces) = finnpy_sr_bem.read_skull_and_skin_models(subj_anatomy_path, subj_name) 
    
    del ws_out_skull_vert; del ws_out_skull_faces
    del ws_out_skin_vect; del ws_out_skin_faces   
    
    in_skull_reduced_vert_path = os.path.join(bem_save_path, 
                                        subj_name+'-'+'in_skull_reduced_vert.npy')
    
    in_skull_faces_path = os.path.join(bem_save_path, 
                                        subj_name+'-'+'in_skull_faces.npy')
    
    in_skull_faces_area_path = os.path.join(bem_save_path, 
                                        subj_name+'-'+'in_skull_faces_area.npy')
    
    in_skull_faces_normal_path = os.path.join(bem_save_path, 
                                        subj_name+'-'+'in_skull_faces_normal.npy')
    
    bem_solution_path = os.path.join(bem_save_path, 
                                        subj_name+'-'+'bem_solution.npy')
    
    if not os.path.exists(in_skull_reduced_vert_path):

        (in_skull_reduced_vert, in_skull_faces,
          in_skull_faces_area, in_skull_faces_normal,
          bem_solution) = finnpy_sr_bem.calc_bem_model_linear_basis(ws_in_skull_vert,
                                                                    ws_in_skull_faces)
                                                                   
        np.save(in_skull_reduced_vert_path, in_skull_reduced_vert)  
        np.save(in_skull_faces_path, in_skull_faces)
        np.save(in_skull_faces_area_path, in_skull_faces_area)
        np.save(in_skull_faces_normal_path, in_skull_faces_normal)   
        np.save(bem_solution_path, bem_solution)

    else:
        in_skull_reduced_vert = np.load(in_skull_reduced_vert_path)
        in_skull_faces = np.load(in_skull_faces_path)
        in_skull_faces_area = np.load(in_skull_faces_area_path)
        in_skull_faces_normal = np.load(in_skull_faces_normal_path)
        bem_solution = np.load(bem_solution_path)
        
                                               
                                                               
    (lh_white_vert, lh_white_faces,
      rh_white_vert, rh_white_faces,
      lh_sphere_vert,
      rh_sphere_vert) = finnpy_sr_utils.read_cortical_models(subj_anatomy_path)                                                    

    octa_model_vert_path = os.path.join(bem_save_path, 
                                        subj_name+'-'+'octa_model_vert.npy')
    
    octa_model_faces_path = os.path.join(bem_save_path, 
                                        subj_name+'-'+'octa_model_faces.npy')

    if not os.path.exists(octa_model_vert_path):
    
        (octa_model_vert, octa_model_faces) = finnpy_sr_smm.create_source_mesh_model()
        np.save(octa_model_vert_path, octa_model_vert)
        np.save(octa_model_faces_path, octa_model_faces)
        
    else:
        octa_model_vert = np.load(octa_model_vert_path)
        octa_model_faces = np.load(octa_model_faces_path)
        
    
    lh_white_valid_vert_path = os.path.join(bem_save_path, 
                                        subj_name+'-'+'lh_white_valid_vert.npy')
    
    rh_white_valid_vert_path = os.path.join(bem_save_path, 
                                        subj_name+'-'+'rh_white_valid_vert.npy')
    
    if not os.path.exists(lh_white_valid_vert_path):
    
        (lh_white_valid_vert,
          rh_white_valid_vert) = finnpy_sr_smm.match_source_mesh_model(lh_sphere_vert,
                                                                      rh_sphere_vert,
                                                                      octa_model_vert)                                             
        np.save(lh_white_valid_vert_path, lh_white_valid_vert)
        np.save(rh_white_valid_vert_path, rh_white_valid_vert)  
    else:
        lh_white_valid_vert = np.load(lh_white_valid_vert_path)
        rh_white_valid_vert = np.load(rh_white_valid_vert_path)
                                                           
                        
    return (lh_white_vert, rh_white_vert, in_skull_reduced_vert,
        in_skull_faces, in_skull_faces_normal, in_skull_faces_area, bem_solution,
        lh_white_valid_vert, rh_white_valid_vert, lh_white_faces, rh_white_faces,
        octa_model_vert, octa_model_faces)                                                      

def recording_specific_processing(subj_name, data_path):
    
    intermediate_save_path = os.path.join(os.path.split(data_path)[0], 'sourceRecIntermediateFiles')
    images_save_path = os.path.join(os.path.split(data_path)[0], 'images')
    
    if not os.path.exists(intermediate_save_path):
        os.mkdir(intermediate_save_path)
    
    if not os.path.exists(images_save_path):
        os.mkdir(images_save_path)
    
    overwrite_mri_trans = False 
    subj_anatomy_path = os.path.join(anatomy_path, subj_name) + '/'
    fiducial_output = os.path.join(subj_anatomy_path, 'bem')
    T1_path = os.path.join(subj_anatomy_path, 'mri', 'T1.mgz')
    
    try:
        if not os.path.exists(T1_path):
            finnpy_sr_mri_anat.copy_fs_avg_anatomy(anatomy_path+'/',
                                             anatomy_path,
                                             subj_name)
            
    
        if not os.path.exists(fiducial_output):
            os.mkdir(fiducial_output)
    
        if not os.path.exists(os.path.join(fiducial_output, subj_name+'-'+'fiducials.fif')):
            _create_fiducials(fs_anatomy_path, subj_anatomy_path, subj_name)
    
        
        rec_meta_info = mne.io.read_info(data_path)
        
        coreg_rotors_free_path = os.path.join(intermediate_save_path,
                                              subj_name+'-'+'coreg_rotors_free.npy')
        
        coreg_rotors_restricted_path = os.path.join(intermediate_save_path,
                                              subj_name+'-'+'coreg_rotors_restricted.npy')
        
        meg_pts_free_path = os.path.join(intermediate_save_path, 
                                         subj_name+'-'+'meg_pts_free.npy')
        
        meg_pts_restricted_path = os.path.join(intermediate_save_path, 
                                         subj_name+'-'+'meg_pts_restricted.npy')
        
        rigid_mri_to_meg_trans_path = os.path.join(intermediate_save_path,
                                                   subj_name+'-'+'rigid_mri_to_meg_trans.npy')
        
        rigid_meg_to_mri_trans_path = os.path.join(intermediate_save_path,
                                                   subj_name+'-'+'rigid_meg_to_mri_trans.npy')
        
        if not os.path.exists(coreg_rotors_free_path):
            
            (coreg_rotors,
              meg_pts) = finnpy_sr_coreg.calc_coreg(subj_name,subj_anatomy_path,
                                                    rec_meta_info,
                                                    registration_scale_type = "free")
                                                    
            np.save(coreg_rotors_free_path, coreg_rotors)
            np.save(meg_pts_free_path, meg_pts, allow_pickle = True)
                                                    
        else:        
            coreg_rotors = np.load(coreg_rotors_free_path)
            meg_pts = np.load(meg_pts_free_path, allow_pickle = True)
                                               
        finnpy_sr_mri_anat.scale_anatomy(subj_anatomy_path, subj_name, coreg_rotors[6:9])
        
        
        if not os.path.exists(coreg_rotors_restricted_path):
            (coreg_rotors,
              meg_pts) = finnpy_sr_coreg.calc_coreg(subj_name, subj_anatomy_path,
                                                    rec_meta_info,
                                                    registration_scale_type = "restricted")
                                                    
            np.save(coreg_rotors_restricted_path, coreg_rotors)
            np.save(meg_pts_restricted_path, meg_pts, allow_pickle = True)
    
        else:    
            coreg_rotors = np.load(coreg_rotors_restricted_path)
            meg_pts = np.load(meg_pts_restricted_path, allow_pickle = True)                                        
        
        if not os.path.exists(rigid_mri_to_meg_trans_path):
            rigid_mri_to_meg_trans = finnpy_sr_coreg.get_rigid_transform(coreg_rotors)
            rigid_meg_to_mri_trans = scipy.linalg.inv(rigid_mri_to_meg_trans)
            
            np.save(rigid_mri_to_meg_trans_path, rigid_mri_to_meg_trans)
            np.save(rigid_meg_to_mri_trans_path, rigid_meg_to_mri_trans)
        else:
            rigid_mri_to_meg_trans = np.load(rigid_mri_to_meg_trans_path)
            rigid_meg_to_mri_trans = np.load(rigid_meg_to_mri_trans_path)
        
    
        (lh_white_vert, rh_white_vert,
          in_skull_reduced_vert, in_skull_faces, 
          in_skull_faces_normal, in_skull_faces_area, 
          bem_solution, lh_white_valid_vert, rh_white_valid_vert,
          lh_white_faces, rh_white_faces, octa_model_vert,
          octa_model_faces) = subject_specific_processing(subj_name)
        
        
        fwd_sol_path = os.path.join(intermediate_save_path, 
                                    subj_name+'-'+'fwd_sol.npy')
        
        lh_white_valid_vert_2_path = os.path.join(intermediate_save_path, 
                                    subj_name+'-'+'lh_white_valid_vert2.npy')
        rh_white_valid_vert_2_path = os.path.join(intermediate_save_path, 
                                    subj_name+'-'+'rh_white_valid_vert2.npy')
        
        
    

        if not os.path.exists(fwd_sol_path):
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
                                                                      
            print(subj_name, 'fwd_sol')

            np.save(fwd_sol_path, fwd_sol)
            np.save(lh_white_valid_vert_2_path, lh_white_valid_vert)
            np.save(rh_white_valid_vert_2_path, rh_white_valid_vert)
                                                                      
        else:
            fwd_sol = np.load(fwd_sol_path)
            lh_white_valid_vert = np.load(lh_white_valid_vert_2_path)
            rh_white_valid_vert = np.load(rh_white_valid_vert_2_path)
                                                                   
        
        optimized_fwd_sol_path = os.path.join(intermediate_save_path, 
                                    subj_name+'-'+'optimized_fwd_sol.npy')    
        
        if not os.path.exists(optimized_fwd_sol_path):                                        
        
            optimized_fwd_sol = finnpy_sr_fwd.optimize_fwd_model(lh_white_vert, lh_white_faces,
                                                                  lh_white_valid_vert, rh_white_vert,
                                                                  rh_white_faces, rh_white_valid_vert,
                                                                  fwd_sol, rigid_mri_to_meg_trans)                                                         
            print(subj_name, 'optimized_fwd_sol')
            
            np.save(optimized_fwd_sol_path,optimized_fwd_sol)
            
        else:
            optimized_fwd_sol = np.load(optimized_fwd_sol_path)
        
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
        
        if not os.path.exists(inv_trans_path):
            (inv_trans, noise_norm) = finnpy_sr_inv.calc_inverse_model(sensor_cov_eigen_val, sensor_cov_eigen_vec,
                                                            sensor_cov_names, optimized_fwd_sol,
                                                            rec_meta_info)
            
            print(subj_name, 'optimized_inv_model')
            
            np.save(inv_trans_path, inv_trans)
            np.save(noise_norm_path, noise_norm)
            
        else:
            inv_trans = np.load(inv_trans_path)
            noise_norm = np.load(noise_norm_path)
            
            
        if not os.path.exists(fs_avg_trans_mat_path):
            
            (fs_avg_trans_mat,
            src_fs_avg_valid_lh_vert,
            src_fs_avg_valid_rh_vert) = finnpy_sr_utils.get_mri_subj_to_fs_avg_trans_mat(lh_white_valid_vert,
                                                                                      rh_white_valid_vert,
                                                                                      octa_model_vert,
                                                                                      subj_anatomy_path, fs_anatomy_path,
                                                                                      overwrite = overwrite_mri_trans)                                                                                         
            
            
            print(subj_name, 'fs_avg_trans_mat')
                                                                             
            save_npz(fs_avg_trans_mat_path, fs_avg_trans_mat)
            np.save(src_fs_avg_valid_lh_vert_path, src_fs_avg_valid_lh_vert)
            np.save(src_fs_avg_valid_rh_vert_path, src_fs_avg_valid_rh_vert)   
                                                                                         
        else:
            fs_avg_trans_mat = load_npz(fs_avg_trans_mat_path)
            src_fs_avg_valid_lh_vert = np.load(src_fs_avg_valid_lh_vert_path)
            src_fs_avg_valid_rh_vert = np.load(src_fs_avg_valid_rh_vert_path)
            

        np.save(octa_model_vert_path, octa_model_vert)
        np.save(octa_model_faces_path, octa_model_faces)
                                                                                     
        return (inv_trans, noise_norm, fs_avg_trans_mat, 
                src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert,
                octa_model_vert, octa_model_faces)
        
    except:
        return (None, None, None, None, None, None, None)  

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
    
    
    
def transform_data(subj_name, data_path):
    
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
    
    morphed_epoch_data_path = os.path.join(intermediate_save_path,
                                    subj_name+'-'+'morphed_epoch_data.npy')
    
    
    morphed_epoch_channels_path = os.path.join(intermediate_save_path,
                                    subj_name+'-'+'morphed_epoch_channels.npy')
    
    morphed_region_names_path = os.path.join(intermediate_save_path,
                                    subj_name+'-'+'morphed_region_names.npy')
    
    if os.path.exists(morphed_epoch_data_path):
        morphed_epoch_data = np.load(morphed_epoch_data_path)
        morphed_epoch_channels = np.load(morphed_epoch_channels_path, allow_pickle=True)
        morphed_region_names = np.load(morphed_region_names_path)
        return (morphed_epoch_data, morphed_epoch_channels, morphed_region_names)
    
    
    try:
        if not os.path.exists(inv_trans_path):        
            recording_specific_processing(subj_name, data_path)
        
        inv_trans = np.load(inv_trans_path)
        noise_norm = np.load(noise_norm_path)
        fs_avg_trans_mat = load_npz(fs_avg_trans_mat_path)
        src_fs_avg_valid_lh_vert = np.load(src_fs_avg_valid_lh_vert_path)
        src_fs_avg_valid_rh_vert = np.load(src_fs_avg_valid_rh_vert_path)
        octa_model_vert = np.load(octa_model_vert_path)
        octa_model_faces = np.load(octa_model_faces_path)
        
            
        raw_data = mne.io.read_raw_fif(data_path, preload=True) 
        if raw_data.times[-1] > 200:
            long_recording = True
        else:
            long_recording = False
            #return (None, None, None)
    
        ch_names = raw_data.info["ch_names"]
        ch_types = __get_ch_types(raw_data.info)
                                                                                 
        sen_data = raw_data.get_data()
        sen_data = __remove_bad_channels(sen_data, ch_names, ch_types)
        
        if long_recording:
            epoched_sen_data = []
            sen_data_length = sen_data.shape[1]
            num_segs = math.ceil(raw_data.times[-1]/200)
            seg_length = math.ceil(sen_data_length/num_segs)
            
            for i in range(num_segs):
                if (i+1)*seg_length >= sen_data_length:
                    epoched_sen_data = sen_data[:,i*seg_length:]
                   
                else:
                   epoched_sen_data = sen_data[:,i*seg_length:(i+1)*seg_length]
                   
                (morphed_epoch_data_i, 
                 morphed_epoch_channels_i, 
                 morphed_region_names_i) = transform_data_matrix(epoched_sen_data, inv_trans, noise_norm, fs_avg_trans_mat, 
                                           src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert,
                                           octa_model_vert, octa_model_faces) 
                
                if i==0:
                    morphed_epoch_data = morphed_epoch_data_i
                    morphed_epoch_channels = morphed_epoch_channels_i
                    morphed_region_names = morphed_region_names_i
                else:
                    morphed_epoch_data = np.concatenate((morphed_epoch_data, morphed_epoch_data_i), axis = 1)
                    
        else:
            (morphed_epoch_data, 
             morphed_epoch_channels, 
             morphed_region_names) = transform_data_matrix(sen_data, inv_trans, noise_norm, fs_avg_trans_mat, 
                                       src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert,
                                       octa_model_vert, octa_model_faces) 
                        
        
        np.save(morphed_epoch_data_path, morphed_epoch_data)
        np.save(morphed_epoch_channels_path, np.array(morphed_epoch_channels, dtype=object), allow_pickle= True)
        np.save(morphed_region_names_path, morphed_region_names)                                           
        return (morphed_epoch_data, morphed_epoch_channels, morphed_region_names)
            
    except:
        return (None, None, None)
        



#Initialize general paths
fs_anatomy_path = '/storage/prerana/subjects/VerifySourceReconstructionPipeline/fsaverage/'

#anatomy_path = '/storage/prerana/subjects/pd/'
#fs_path = '/usr/local/freesurfer/7.4.1-1/'

anatomy_path = '/mnt/VerifySourceReconstructionPipeline'
fs_path =  '/usr/local/freesurfer/7.4.1/'

sensor_cov_path = '/storage/prerana/sensor_covariance_file.fif'
cov_path = '/storage/prerana/sensor_covariance_file_processed'

overwrite_sensor_cov = False


all_subj_dirs = os.listdir(anatomy_path)

if 'fsaverage' in all_subj_dirs:
    all_subj_dirs.remove('fsaverage')
if 'al0008a' in all_subj_dirs:
    all_subj_dirs.remove('al0008a')
non_recording_folders = ['bem', 'label', 'mri', 'proj', 'scripts', 'stats', 'surf', 'tmp', 'touch', 'trash', '.is_scaled']

param_list = []

#Preparing parameter list for multiprocessing (each combo of subject name and respective recording files)
# for subj_name in all_subj_dirs:
#     subj_specific_path = os.path.join(anatomy_path, subj_name)
#     subj_subfolders = os.listdir(subj_specific_path)
    
    
#     if len(subj_subfolders) != 0:
#         data_folders = [folder for folder in subj_subfolders if folder not in non_recording_folders ]

#     if len(data_folders) != 0:
#         data_path = os.path.join(subj_specific_path, data_folders[0], 
#                                   data_folders[0]+'.fif')
        
#         parameter_tuple = (subj_name, data_path)
#         param_list.append(parameter_tuple)
#     else:
#         continue
    
# for subj_name in all_subj_dirs:
#     subj_specific_path = os.path.join(anatomy_path, subj_name)
#     subj_subfolders = os.listdir(subj_specific_path)
    
#     if len(subj_subfolders) != 0:
#         data_folders = [folder for folder in subj_subfolders if folder not in non_recording_folders ]
    
    
#     if len(data_folders) > 0:
#         for i in range (len(data_folders)):
            
#             data_path = os.path.join(subj_specific_path, data_folders[i], 
#                                      data_folders[i]+'.fif')
            
#             morphed_data_path = os.path.join(subj_specific_path, data_folders[i], 'sourceRecIntermediateFiles',
#                                      subj_name+'-'+'morphed_epoch_data.npy')            
            
            
#             parameter_tuple = (subj_name, data_path)
#             if not os.path.exists(morphed_data_path):
#                 param_list.append(parameter_tuple)




#Subject agnostic functions
finnpy_sr_utils.init_fs_paths(anatomy_path, fs_path)

(sensor_cov_eigen_val,
 sensor_cov_eigen_vec,
 sensor_cov_names) = finnpy_sr_sc.get_sensor_covariance(file_path = sensor_cov_path, 
                                                        cov_path = cov_path,
                                                        overwrite = overwrite_sensor_cov)



#al0008a don't use

thread_cnt = 7
function = recording_specific_processing
#function = transform_data

#idxs = [0,1,3,4]
#old_param_list = param_list
#param_list = []

#for i in range(8,8+thread_cnt):
#    param_list.append(old_param_list[i])

param_list = [('test001', '/storage/prerana/subjects/VerifySourceReconstructionPipeline/test001/meta_data_only/meta_data_only.fif')]

#param_list = param_list[317:] #314 being tricky
# param_list = param_list[2:]
for params in param_list:
    t1 = time.perf_counter()
    
    #recording_specific_processing(param_list[1][0],param_list[1][1])
    #function(*param_list[2])
    data = function(*params)
    # del data
    #transform_data(*param_list[i])
    #param_list = param_list[1:]
    #transformation_matrices = tp.run(thread_cnt, function, param_list)
    #function(*param_list[0])
    
    
    t2 = time.perf_counter()
    
    runttime = t2-t1
    print('Runtime:', runttime)