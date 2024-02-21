#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:56:22 2024

@author: prerana
"""

import os

import mne
import time
import scipy
from scipy.sparse import save_npz, load_npz
import numpy as np

#Initialize general paths
fs_anatomy_path = '/storage/prerana/subjects/pd/fsaverage/'
anatomy_path = '/storage/prerana/subjects/pd/'
fs_path = '/usr/local/freesurfer/7.4.1-1/'


all_subj_dirs = os.listdir(anatomy_path)
all_subj_dirs.remove('fsaverage')
non_recording_folders = ['bem', 'label', 'mri', 'proj', 'scripts', 'stats', 'surf', 'tmp', 'touch', 'trash', '.is_scaled']

param_list = []


for subj_name in all_subj_dirs:
    subj_specific_path = os.path.join(anatomy_path, subj_name)
    subj_subfolders = os.listdir(subj_specific_path)
    
    
    if len(subj_subfolders) != 0:
        data_folders = [folder for folder in subj_subfolders if folder not in non_recording_folders ]
        

    if len(data_folders) != 0:    
        for i in range (len(data_folders)):
            morphed_data_path = os.path.join(subj_specific_path, data_folders[i], 'sourceRecIntermediateFiles',
                                     subj_name+'-'+'morphed_epoch_data.npy')
            
            fs_avg_src_data_path = os.path.join(subj_specific_path, data_folders[i], 'sourceRecIntermediateFiles',
                                     subj_name+'-'+'fs_avg_src_data.npy')
                                     
            src_data_path = os.path.join(subj_specific_path, data_folders[i], 'sourceRecIntermediateFiles',
                                     subj_name+'-'+'src_data.npy')
            
            if not os.path.exists(morphed_data_path):
                param_list.append(os.path.join(subj_specific_path, data_folders[i]))
            
            if os.path.exists(fs_avg_src_data_path):
                print(fs_avg_src_data_path)
                os.remove(fs_avg_src_data_path)
                
            if os.path.exists(src_data_path):
                print(src_data_path)
                os.remove(src_data_path)

    else:
        continue