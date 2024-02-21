#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:34:22 2024

@author: prerana
"""

import numpy as np
import os
import random

import matplotlib
#matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import finnpy.cleansing.bad_channel_identification as bci
import finnpy.cleansing.channel_restoration as cr
import mne

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

def read_data(data_path):
    raw_data = mne.io.read_raw_fif(data_path, preload=True) 
    ch_names = raw_data.info["ch_names"]
    ch_types = __get_ch_types(raw_data.info)
                                                                             
    sen_data = raw_data.get_data()
    sen_data, ch_names = __remove_bad_channels(sen_data, ch_names, ch_types)
    fs = raw_data.info['sfreq']
    
    return sen_data, ch_names, fs
    


def channel_cleanse(data_path):
    intermediate_save_path = os.path.join(os.path.split(data_path)[0], 'sourceRecIntermediateFiles')
    
    #Configure sample data
    raw_data, ch_names, fs = read_data(data_path)
    channel_count = len(ch_names)

    #Faulty channel gets identified
    (_, invalid_list, _) = bci.run(raw_data, ch_names, [fs for _ in range(channel_count)], [[60, 100]], broadness = 3, visual_inspection = False)
    #Faulty channel gets substituted via neighbors
    rest_data = cr.run(raw_data, ch_names, invalid_list)

    #visualization
    if len(invalid_list) > 0:
        channels_to_plot = 3
        (_, axes) = plt.subplots(channels_to_plot, 2)
        for channel_idx in range(channels_to_plot):
            axes[channel_idx, 0].plot(raw_data[channel_idx][:200])
            axes[channel_idx, 1].plot(rest_data[channel_idx][:200])
    
        axes[0, 0].set_title("before correction")
        axes[0, 1].set_title("after correction")
    
        axes[0, 0].set_ylabel("Channel #0\n"); #axes[0, 0].set_yticks([-2, 0, 2])
        axes[1, 0].set_ylabel("Channel #1\n(faulty channel)"); #axes[1, 0].set_yticks([-2, 0, 2])
        axes[2, 0].set_ylabel("Channel #2\n"); #axes[2, 0].set_yticks([-2, 0, 2])
    
        plt.show()
        
        return (data_path, invalid_list)
    
    else:
        return (None, None)


#Initialize general paths
anatomy_path = '/storage/prerana/subjects/pd/'
fs_path = '/usr/local/freesurfer/7.4.1-1/'


all_subj_dirs = os.listdir(anatomy_path)
all_subj_dirs.remove('fsaverage')
all_subj_dirs.remove('al0008a')
non_recording_folders = ['bem', 'label', 'mri', 'proj', 'scripts', 'stats', 'surf', 'tmp', 'touch', 'trash', '.is_scaled']

path_list = []
faulty_sensors_files = []
faulty_sensors = []

for subj_name in all_subj_dirs:
    subj_specific_path = os.path.join(anatomy_path, subj_name)
    subj_subfolders = os.listdir(subj_specific_path)
    
    if len(subj_subfolders) != 0:
        data_folders = [folder for folder in subj_subfolders if folder not in non_recording_folders ]
    
    
    if len(data_folders) > 0:
        for i in range (len(data_folders)):
            
            data_path = os.path.join(subj_specific_path, data_folders[i], 
                                     data_folders[i]+'.fif')
            
            (data_path, invalid_list) = channel_cleanse(data_path)
            
            if data_path is None:
                continue
            else:
                faulty_sensors_files.append(data_path)
                faulty_sensors.append(invalid_list)
            
            path_list.append(data_path)

