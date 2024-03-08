#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:55:33 2024

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

import os
import finnpy.source_reconstruction.utils
import shutil

import nibabel.freesurfer
import numpy as np
import finnpy.source_reconstruction.utils as finnpy_utils

import mne.io

def copy_anatomy(original_path, new_path, subj_name):
   
    
    old_base_dir = os.path.join(original_path, subj_name) + '/'
    new_base_dir = os.path.join(new_path, subj_name) + '/'
    
    if not (os.path.exists(new_base_dir)):
        os.mkdir(new_base_dir)
     
    #Create and populate bem folder
    bem_dir = os.path.join(new_base_dir, 'bem')
    if not os.path.exists(bem_dir):
        os.mkdir(bem_dir)
    shutil.copyfile(old_base_dir + "bem/" + subj_name + "-fiducials.fif_unscaled", new_base_dir + "bem/" + subj_name + "-fiducials.fif")
    #shutil.copyfile(mne.__file__[:mne.__file__.rindex("/")] + "/data/fsaverage/fsaverage-fiducials.fif", new_base_dir + "bem/" + subj_name + "-fiducials.fif")
    
    watershed_dir = os.path.join(bem_dir, 'watershed')
    if not os.path.exists(watershed_dir):
        os.mkdir(watershed_dir) 
    
    #Create and populate mri folder
    
    mri_dir = os.path.join(new_base_dir, 'mri')
    if not os.path.exists(mri_dir):
        os.mkdir(mri_dir)
    
    transforms_dir = os.path.join(mri_dir, 'transforms')
    if not os.path.exists(transforms_dir):
        os.mkdir(transforms_dir)
    
    shutil.copyfile(old_base_dir + "mri/" + "orig.mgz_unscaled", new_base_dir + "mri/" + "orig.mgz")
    shutil.copyfile(old_base_dir + "mri/" + "T1.mgz_unscaled", new_base_dir + "mri/" + "T1.mgz")
    shutil.copyfile(old_base_dir + "mri/transforms/" + "talairach.xfm", new_base_dir + "mri/transforms/" + "talairach.xfm")
     
    #Create and populate surface folder
    surf_dir = os.path.join(new_base_dir, 'surf')
    if not os.path.exists(surf_dir):
        os.mkdir(surf_dir) 
        
    shutil.copyfile(old_base_dir + "surf/" + "lh.sphere", new_base_dir + "surf/" + "lh.sphere")
    shutil.copyfile(old_base_dir + "surf/" + "lh.sphere.reg", new_base_dir + "surf/" + "lh.sphere.reg")
    shutil.copyfile(old_base_dir + "surf/" + "lh.white_unscaled", new_base_dir + "surf/" + "lh.white")
    shutil.copyfile(old_base_dir + "surf/" + "rh.sphere", new_base_dir + "surf/" + "rh.sphere")
    shutil.copyfile(old_base_dir + "surf/" + "rh.sphere.reg", new_base_dir + "surf/" + "rh.sphere.reg")
    shutil.copyfile(old_base_dir + "surf/" + "rh.white_unscaled", new_base_dir + "surf/" + "rh.white")
    

def check_coregistration_accuracy(subj_name, data_path, original_anatomy_path, new_anatomy_path):
    subj_anatomy_path = new_anatomy_path + subj_name + '/'
    
    
    copy_anatomy(original_anatomy_path, new_anatomy_path, subj_name)
    
    intermediate_save_path = os.path.join(os.path.split(data_path)[0], 'sourceRecIntermediateFiles')
    rec_meta_info = mne.io.read_info(data_path)
        
    coreg_rotors_free_path = os.path.join(intermediate_save_path,
                                          subj_name+'-'+'coreg_rotors_free.npy')
    
    coreg_rotors_restricted_path = os.path.join(intermediate_save_path,
                                          subj_name+'-'+'coreg_rotors_restricted.npy')
    
    meg_pts_free_path = os.path.join(intermediate_save_path, 
                                     subj_name+'-'+'meg_pts_free.npy')
    
    meg_pts_restricted_path = os.path.join(intermediate_save_path, 
                                     subj_name+'-'+'meg_pts_restricted.npy')
    
   
    (coreg_rotors_free_new,
      meg_pts_free_new) = finnpy_sr_coreg.calc_coreg(subj_name,subj_anatomy_path,
                                            rec_meta_info,
                                            registration_scale_type = "free")
                                                
                  
    coreg_rotors_free_old = np.load(coreg_rotors_free_path)
    meg_pts_free_old = np.load(meg_pts_free_path, allow_pickle = True)
                                               
    finnpy_sr_mri_anat.scale_anatomy(subj_anatomy_path, subj_name, coreg_rotors_free_new[6:9])
        
    (coreg_rotors_restricted_new,
      meg_pts_restricted_new) = finnpy_sr_coreg.calc_coreg(subj_name, subj_anatomy_path,
                                            rec_meta_info,
                                            registration_scale_type = "restricted")
                                                    

    coreg_rotors_restricted_old = np.load(coreg_rotors_restricted_path)
    meg_pts_restricted_old = np.load(meg_pts_restricted_path, allow_pickle = True)       

    match = np.all(coreg_rotors_restricted_new == coreg_rotors_restricted_old)    
    
    shutil.rmtree(subj_anatomy_path)
    
    return (data_path, match)                           
        


original_path = '/storage/prerana/subjects/dy/'
fs_path = '/usr/local/freesurfer/7.4.1-1/'
original_subjects = os.listdir(original_path)
if 'fsaverage' in original_subjects:
    original_subjects.remove('fsaverage')
if 'al0008a' in original_subjects:
    original_subjects.remove('al0008a')

redo_path = '/storage/prerana/subjects_redo/dy/'

finnpy_sr_utils.init_fs_paths(redo_path, fs_path)


non_recording_folders = ['bem', 'label', 'mri', 'proj', 'scripts', 'stats', 'surf', 'tmp', 'touch', 'trash', '.is_scaled']


param_list = []
#original_subjects = ['al0069b']
for subj_name in original_subjects:
    subj_specific_path = os.path.join(original_path, subj_name)
    subj_subfolders = os.listdir(subj_specific_path)
    
    if len(subj_subfolders) != 0:
        data_folders = [folder for folder in subj_subfolders if folder not in non_recording_folders ]
    
    
    if len(data_folders) > 0:
        for i in range (len(data_folders)):
            
            data_path = os.path.join(subj_specific_path, data_folders[i], 
                                      data_folders[i]+'.fif')
            
            morphed_data_path = os.path.join(subj_specific_path, data_folders[i], 'sourceRecIntermediateFiles',
                                      subj_name+'-'+'morphed_epoch_data.npy')            
            
            
            parameter_tuple = (subj_name, data_path)
            param_list.append(parameter_tuple)

recheck_files = []
for params in param_list:
    t1 = time.perf_counter()
    flag = check_coregistration_accuracy(*params, original_path, redo_path)
    
    if not flag:
        recheck_files.append(params)
    
    t2 = time.perf_counter()
    
    runttime = t2-t1
    print('Runtime:', runttime)
