#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:42:49 2024

@author: prerana
"""

import os
import glob 
import subprocess

FREESURFER_HOME = '/usr/local/freesurfer/7.4.1-1/subjects'
all_nii_files = []
failed_processing = []

# os.chdir('/mnt/patient_data/pd_nifti')
# storage_dir = '/storage/prerana/subjects/pd'
# path = os.getcwd()
# dir_list = os.listdir(path)



# #dir_list.remove('al00103a')\
# already_processed = os.listdir(storage_dir)
# for processed_folder in already_processed:
#     if processed_folder in dir_list:
#         dir_list.remove(processed_folder)

# for directory in dir_list:
#     if directory[-1] == 'b':
#         if directory in dir_list:
#             dir_list.remove(directory)

# for directory in dir_list:
#     try:
#         folder_path = os.path.join(path, directory)
#         os.chdir(folder_path)
#         MRI_files = glob.glob('*.nii.gz')
        
#         if len(MRI_files) == 0:
#             print('Cannot find MRI (.nii.gz) extension in: ', directory)
#             continue
#         else:
#             MRI_file = MRI_files[0]    
#             all_nii_files.append(MRI_file)
        
            
#         fullfile = os.path.join(folder_path, MRI_file)
            
#         subprocess.run(['recon-all', '-s', directory, '-i', fullfile, '-all', '-threads', '30'])
#         #Move relevant folders from freesurfer home to storage directory
        
#         freesurfer_subj_dir = os.path.join(FREESURFER_HOME, directory)
        
        
#         #subprocess.run(['mv', freesurfer_subj_dir, storage_dir])
        
#     except:
#         print('Error thrown at: '. directory)
#         failed_processing.append(directory)
#         continue
    
    
    
# os.chdir('/mnt/patient_data/dy_nifti')
# FREESURFER_HOME = '/usr/local/freesurfer/7.4.1-1/subjects'
# storage_dir = '/storage/prerana/subjects/dy'
# path = os.getcwd()
# dir_list = os.listdir(path)



# #dir_list.remove('al00103a')\
    
# already_processed = os.listdir(storage_dir)
# for processed_folder in already_processed:
#     dir_list.remove(processed_folder)

# for directory in dir_list:
#     if directory[-1] == 'b':
#         dir_list.remove(directory)

# for directory in dir_list:
#     try:
#         folder_path = os.path.join(path, directory)
#         os.chdir(folder_path)
#         MRI_files = glob.glob('*.nii.gz')
        
#         if len(MRI_files) == 0:
#             print('Cannot find MRI (.nii.gz) extension in: ', directory)
#             continue
#         else:
#             MRI_file = MRI_files[0]    
#             all_nii_files.append(MRI_file)
        
            
#         fullfile = os.path.join(folder_path, MRI_file)
            
#         subprocess.run(['recon-all', '-s', directory, '-i', fullfile, '-all', '-threads', '30'])
#         #Move relevant folders from freesurfer home to storage directory
        
#         freesurfer_subj_dir = os.path.join(FREESURFER_HOME, directory)
        
        
#         #subprocess.run(['mv', freesurfer_subj_dir, storage_dir])
        
#     except:
#         print('Error thrown at: '. directory)
#         failed_processing.append(directory)
#         continue
    

os.chdir('/mnt/patient_data/et_nifti')
FREESURFER_HOME = '/usr/local/freesurfer/7.4.1-1/subjects'
storage_dir = '/storage/prerana/subjects/et'
path = os.getcwd()
dir_list = os.listdir(path)



#dir_list.remove('al00103a')\
    
already_processed = os.listdir(storage_dir)
for processed_folder in already_processed:
    dir_list.remove(processed_folder)

for directory in dir_list:
    if directory[-1] == 'b':
        dir_list.remove(directory)

for directory in dir_list:
    try:
        folder_path = os.path.join(path, directory)
        os.chdir(folder_path)
        MRI_files = glob.glob('*.nii.gz')
        
        if len(MRI_files) == 0:
            print('Cannot find MRI (.nii.gz) extension in: ', directory)
            continue
        else:
            MRI_file = MRI_files[0]    
            all_nii_files.append(MRI_file)
        
            
        fullfile = os.path.join(folder_path, MRI_file)
            
        subprocess.run(['recon-all', '-s', directory, '-i', fullfile, '-all', '-threads', '30'])
        #Move relevant folders from freesurfer home to storage directory
        
        freesurfer_subj_dir = os.path.join(FREESURFER_HOME, directory)
        
        
        #subprocess.run(['mv', freesurfer_subj_dir, storage_dir])
        
    except:
        print('Error thrown at: '. directory)
        failed_processing.append(directory)
        continue
    
    

