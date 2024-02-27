#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:19:51 2024

@author: prerana
"""


import finnpy.source_reconstruction.coregistration_meg_mri as finnpy_sr_coreg
import finnpy.source_reconstruction.bem_model as finnpy_sr_bem
import nibabel.freesurfer
import scipy.linalg
import mayavi.mlab
import os
import numpy as np
import mne




def plot_coregistration(coreg, rec_meta_info, meg_pts, subj_path, data_path):
    """
    Plots the result of the coregistration from MRI to MEG using mayavi.
    
    Parameters
    ----------
    coreg : numpy.ndarray, shape(4, 4)
            MEG to MRI coregistration matrix.
    rec_meta_info : mne.io.read_info
                    MEG scan meta information, obtainable through mne.io.read_info.
    meg_pts : numpy.ndarray, shape(n, 4)
              MEG pts used in the coregistration.
    subj_path : string
                Path to the subject's freesurfer files.
    """
    
    (vert, faces) = nibabel.freesurfer.read_geometry(subj_path + "/surf/lh.seghead")
    vert/= 1000
    
    _ = mayavi.mlab.figure(size = (800, 800))
    mayavi.mlab.triangular_mesh(vert[:, 0], vert[:, 1], vert[:, 2], faces, color = (.4, .4, .4), opacity = 0.9)
    
    meg_nasion = np.expand_dims(np.asarray(meg_pts["nasion"]), axis = 0)
    meg_lpa = np.expand_dims(np.asarray(meg_pts["lpa"]), axis = 0)
    meg_rpa = np.expand_dims(np.asarray(meg_pts["rpa"]), axis = 0)
    meg_hpi = np.asarray(meg_pts["hpi"])
    meg_hsp = np.asarray(meg_pts["hsp"])
    
    def meg_to_head_trans(trans, meg_nasion, meg_lpa, meg_rpa, meg_hpi, meg_hsp):
        
        inv_trans = scipy.linalg.inv(trans)
        
        meg_nasion  = np.dot(meg_nasion, inv_trans[:3, :3].T);  meg_nasion  += inv_trans[:3, 3]
        meg_lpa     = np.dot(meg_lpa, inv_trans[:3, :3].T);     meg_lpa     += inv_trans[:3, 3]
        meg_rpa     = np.dot(meg_rpa, inv_trans[:3, :3].T);     meg_rpa     += inv_trans[:3, 3]
        meg_hpi     = np.dot(meg_hpi, inv_trans[:3, :3].T);     meg_hpi     += inv_trans[:3, 3]
        meg_hsp     = np.dot(meg_hsp, inv_trans[:3, :3].T);     meg_hsp     += inv_trans[:3, 3]
        
        return (meg_nasion, meg_lpa, meg_rpa, meg_hpi, meg_hsp)
    
    (meg_nasion, meg_lpa, meg_rpa, meg_hpi, meg_hsp) = meg_to_head_trans(coreg, meg_nasion, meg_lpa, meg_rpa, meg_hpi, meg_hsp)
    
    def mri_to_head_trans(trans, vert):
        
        inv_trans = scipy.linalg.inv(trans)
        
        vert = np.dot(vert, inv_trans[:3, :3].T); vert += inv_trans[:3, 3]
        
        return vert
    
    vert = mri_to_head_trans(rec_meta_info["dev_head_t"]["trans"], vert)
    
    mayavi.mlab.points3d(meg_nasion[:, 0], meg_nasion[:, 1], meg_nasion[:, 2], scale_factor = .015, color = (1, 0, 0))
    mayavi.mlab.points3d(meg_lpa[:, 0], meg_lpa[:, 1], meg_lpa[:, 2], scale_factor = .015,  color = (1, 0.425, 0))
    mayavi.mlab.points3d(meg_rpa[:, 0], meg_rpa[:, 1], meg_rpa[:, 2], scale_factor = .015,  color = (1, 0.425, 0))
    mayavi.mlab.points3d(meg_hpi[:, 0], meg_hpi[:, 1], meg_hpi[:, 2], scale_factor = .01,   color = (1, 0.8, 0))
    mayavi.mlab.points3d(meg_hsp[:, 0], meg_hsp[:, 1], meg_hsp[:, 2], scale_factor = .0025, color = (1, 1, 0))
    
    #mayavi.mlab.show()
    #save_path = os.path.join(intermediate_save_path, 'coregistration_plot.oogl')
    #mayavi.mlab.savefig(save_path)





def visualize_coregistration(subj_name, data_path):
    parsed_data_path = os.path.split(data_path)[0]
    subj_anatomy_path = os.path.split(parsed_data_path)[0] 
    anatomy_path = os.path.split(subj_anatomy_path)[0] 
    subj_anatomy_path = subj_anatomy_path + '/'
    anatomy_path = anatomy_path + '/'
    intermediate_save_path = os.path.join(parsed_data_path, 'sourceRecIntermediateFiles')
    
    coreg_rotors_restricted_path = os.path.join(intermediate_save_path,
                                          subj_name+'-'+'coreg_rotors_restricted.npy')
    
    meg_pts_restricted_path = os.path.join(intermediate_save_path, 
                                     subj_name+'-'+'meg_pts_restricted.npy')
    
    rigid_mri_to_meg_trans_path = os.path.join(intermediate_save_path,
                                               subj_name+'-'+'rigid_mri_to_meg_trans.npy')
    
    rec_meta_info = mne.io.read_info(data_path)

    
    coreg_rotors = np.load(coreg_rotors_restricted_path)
    rigid_mri_to_meg_trans = np.load(rigid_mri_to_meg_trans_path)
    meg_pts = np.load(meg_pts_restricted_path, allow_pickle = True).item()
    
    
    rigid_mri_to_meg_trans = finnpy_sr_coreg.get_rigid_transform(coreg_rotors)
    plot_coregistration(rigid_mri_to_meg_trans, rec_meta_info, 
                                        meg_pts, subj_anatomy_path, data_path)



def generate_skull_skin_plots(subj_name,anatomy_path):
    subj_anatomy_path = os.path.join(anatomy_path, subj_name) + '/'
    
    (ws_in_skull_vert,
     ws_in_skull_faces,
     ws_out_skull_vert,
     ws_out_skull_faces,
     ws_out_skin_vect,
     ws_out_skin_faces) = finnpy_sr_bem.read_skull_and_skin_models(subj_anatomy_path,
                                                                   subj_name)

    finnpy_sr_bem.plot_skull_and_skin_models(ws_in_skull_vert,
                                               ws_in_skull_faces,
                                               ws_out_skull_vert,
                                               ws_out_skull_faces,
                                               ws_out_skin_vect,
                                               ws_out_skin_faces,
                                               subj_anatomy_path)


def get_all_recording_files(anatomy_path):
    all_subj_dirs = os.listdir(anatomy_path)
    if 'fsaverage' in all_subj_dirs:
        all_subj_dirs.remove('fsaverage')
    if 'al0008a' in all_subj_dirs:
        all_subj_dirs.remove('al0008a')
    
    non_recording_folders = ['bem', 'label', 'mri', 'proj', 'scripts', 'stats', 'surf', 'tmp', 'touch', 'trash', '.is_scaled']
    
    param_list_file = []
    param_list_subj = []
    
    for subj_name in all_subj_dirs:
        subj_specific_path = os.path.join(anatomy_path, subj_name)
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
                if os.path.exists(morphed_data_path):
                    param_list_file.append(parameter_tuple)
                    
        subj_tuple = (subj_name, anatomy_path)
        param_list_subj.append(subj_tuple)
        
    return param_list_file, param_list_subj


anatomy_path = '/storage/prerana/subjects/'
conditions = ['pd', 'dy', 'et']

param_list_file = []
param_list_subj = []
for condition in conditions:
    full_anatomy_path = os.path.join(anatomy_path, condition)
    file_tuples, subj_tuples =  get_all_recording_files(full_anatomy_path)
    param_list_file = param_list_file+file_tuples
    param_list_subj = param_list_subj+subj_tuples


# for params in param_list_subj: 
#     print(*params)
#     generate_skull_skin_plots(*params)


for params in param_list_file:
    print(*params)
    visualize_coregistration(*params) #Put a break point here
    
