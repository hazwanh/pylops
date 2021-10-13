#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:43:37 2021

@author: csi-13
"""

import segyio
import matplotlib.pyplot as plt
from scipy import io

#%%
filename = '/home/csi-13/Downloads/ModelParams/Theta_Model.sgy'
with segyio.open(filename,ignore_geometry=True) as f:
  # Get basic attributes
  n_traces = f.tracecount
  sample_rate = segyio.tools.dt(f)/1000
  n_samples = f.samples.size
  data = f.trace.raw[:] # get all data into memory
  print('N Traces:', n_traces,' N Samples:', n_samples,' Sample rate:', sample_rate,'ms')
  
#%%
plt.figure(figsize=(15,5))
plt.imshow(data.T,cmap='jet')
plt.xlabel('Distance (m)')
plt.ylabel('Time (ms)')
plt.show()

#%%
theta_0z_2000z_6500x_9500x = data.T[:2000,6500:9500]
plt.figure()
plt.imshow(theta_0z_2000z_6500x_9500x,cmap='jet')
plt.xlabel('Distance (m)')
plt.ylabel('Time (ms)')
plt.show()

#%%
theta_0z_1000z_4500x_6000x = data.T[:1000,4500:6000]
plt.figure()
plt.imshow(theta_0z_1000z_4500x_6000x,cmap='jet')
plt.xlabel('Distance (m)')
plt.ylabel('Time (ms)')
plt.show()

#%% Save the crop vp model
io.savemat('BP_Theta_Model.mat',{'theta_0z_2000z_6500x_9500x':theta_0z_2000z_6500x_9500x,
                                   'theta_0z_1000z_4500x_6000x':theta_0z_1000z_4500x_6000x,'dz':sample_rate})

#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import seischan as scp

d_0z_1000z_4500x_6000x = io.loadmat('BP_Vp_Model.mat')['d_0z_1000z_4500x_6000x']



