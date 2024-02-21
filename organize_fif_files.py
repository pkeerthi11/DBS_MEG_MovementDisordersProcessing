#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:32:19 2024

@author: prerana
"""

#Organizing .fif files into appropriate folder structure for subsequence processing

import os
import glob
import subprocess

pd_harddrive_path = '/mnt/patient_data/dy'
pd_storage_path = '/storage/prerana/subjects/dy'

pd_harddrive_files = os.listdir(pd_harddrive_path)
pd_storage_files = os.listdir(pd_storage_path)

#.fif not saved in tsss folder 
#not_in_tsss = ['al0013a','al0021a','al0021b'] #pd
#skip_empty = ['al0060a'] #pd



#All et are fine
not_in_tsss = []
skip_empty = []

not_in_tsss = ['al0059a'] #dy

failed_copy = []
for folder in pd_harddrive_files:
    if folder in not_in_tsss:
        fullpath_source_fif = os.path.join(pd_harddrive_path, folder, 'meg')
    else:
        fullpath_source_fif = os.path.join(pd_harddrive_path, folder, 'meg', 'tsss')
    
    os.chdir(fullpath_source_fif)
    fif_files = glob.glob("*.fif")
    
    for file in fif_files:
        filename = file.split(".fif")[0]
        destination_path = os.path.join(pd_storage_path, folder, filename)
        source_path = os.path.join(fullpath_source_fif, file)
        
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)
        
        try:
            subprocess.run(['cp', source_path, destination_path])
        except:
            failed_copy.append(folder+"/"+file)
        
        
        