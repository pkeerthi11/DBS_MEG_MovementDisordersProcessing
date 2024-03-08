#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:58:07 2024

@author: prerana
"""

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
from finnpy.basic import downsampling as ds


from matplotlib import pyplot as plt
from scipy.signal import welch, butter, filtfilt

def preprocess(data, src_freq, target_freq):
    data_rms = math.sqrt((data**2).mean())
    data_scaled = data/data_rms


    b,a = butter(N=4, Wn=120, btype='low', output='ba', fs=src_freq)

    data_filtered = filtfilt(b, a, data_scaled[0,:])

    data_downsampled = ds.run(data_filtered, src_freq, target_freq)
    
    epochs = window_epochs(data_downsampled, stepsize=1, width=3)
    
    return epochs
    

def window_epochs(data, stepsize=1, width=3):
    return np.hstack( data[i:1+i-width or None:stepsize] for i in range(0,width) )

src_freq = 1000
target_freq = 240

data_path = '/storage/prerana/subjects/pd/al0014a/05_DBSJM_15_9M10P_020_90_tsss_mc/sourceRecIntermediateFiles/al0014a-morphed_epoch_data.npy'

data = np.load(data_path)

epochs = preprocess(data)


f, Pxx = welch(epochs[0], fs=target_freq, window='hann', noverlap=0.5)
plt.semilogy(f, Pxx)


