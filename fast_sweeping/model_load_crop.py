#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:05:30 2021

@author: csi-13
"""

import segyio
import matplotlib.pyplot as plt
from scipy import io
import numpy as np

#%%
filename = '/home/csi-13/Downloads/ModelParams/Epsilon_Model.sgy'
with segyio.open(filename,ignore_geometry=True) as f:
  # Get basic attributes
  n_traces = f.tracecount
  sample_rate = segyio.tools.dt(f)/1000
  n_samples = f.samples.size
  data = f.trace.raw[:] # get all data into memory
  print('N Traces:', n_traces,' N Samples:', n_samples,' Sample rate:', sample_rate,'ms')

# for 1500x1500
# dx = dz = 4;
# model_vp1_int4 = data.T[:1500:dz,4500:6000:dx]
# x1 = np.arange(0,data.T[0:1500,4500:6000].shape[0],4)
# z1 = np.arange(0,data.T[0:1500,4500:6000].shape[1],4)
# x = np.arange(0,model_vp1_int4.shape[0])
# z = np.arange(0,model_vp1_int4.shape[1])

# for 1500x3000
dx = dz = 4;
model_vp1_int4 = data.T[:1500:dx,3750:6750:dz]
x1 = np.arange(0,data.T[:1500,3750:6750].shape[1],4)
z1 = np.arange(0,data.T[:1500,3750:6750].shape[0],4)
x = np.arange(0,model_vp1_int4.shape[1])
z = np.arange(0,model_vp1_int4.shape[0])

plt.figure(figsize=(10,5))
plt.imshow(model_vp1_int4,cmap='jet')
plt.xlabel('Distance (m)')
plt.ylabel('Time (ms)')
plt.title('Vp Model Crop')
plt.show()

#%%

vel_true = model_vp1_int4
x = np.arange(0,np.max(x1)-np.min(x1)+4,4)
z = np.arange(0,np.max(z1)-np.min(z1)+4,4)
nx, nz = len(x1), len(z1)

# Smooth velocity
v0 = 1492 # initial velocity
nsmooth=30

# Receivers
nr = 3
rx = np.linspace(dx*25, (nx-25)*dx, nr)
# rx = np.linspace(dx, (nx)*dx, nr)
rz = 20*np.ones(nr)
recs = np.vstack((rx, rz))
dr = recs[0,1]-recs[0,0]

# Sources
ns = 3
sx = np.linspace(dx*25, (nx-25)*dx, ns)
# sx = np.linspace(dx, (nx)*dx, ns)
sz = 20*np.ones(ns)
sources = np.vstack((sx, sz))
ds = sources[0,1]-sources[0,0]

velmin = 1492
velmax = np.abs(-1*vel_true).max()

plt.figure(figsize=(10,5))
im = plt.imshow(vel_true, cmap='jet', vmin = velmin, vmax = velmax,
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Velocity')
plt.ylim(z[-1], z[0])

#%%
filename = '/home/csi-13/Downloads/ModelParams/Theta_Model.sgy'
with segyio.open(filename,ignore_geometry=True) as f:
  # Get basic attributes
  n_traces = f.tracecount
  sample_rate = segyio.tools.dt(f)/1000
  n_samples = f.samples.size
  data = f.trace.raw[:] # get all data into memory
  print('N Traces:', n_traces,' N Samples:', n_samples,' Sample rate:', sample_rate,'ms')

dx = dz = 4;
model_thet1_int4 = data.T[:1500:dz,4500:6000:dx]
x = np.arange(0,model_vp1_int4.shape[0])
z = np.arange(0,model_vp1_int4.shape[1])

plt.figure(figsize=(15,5))
plt.imshow(model_thet1_int4,cmap='jet')
plt.xlabel('Distance (m)')
plt.ylabel('Time (ms)')
plt.title('Theta Model Crop')
plt.show()

#%%
filename = '/home/csi-13/Downloads/ModelParams/Epsilon_Model.sgy'
with segyio.open(filename,ignore_geometry=True) as f:
  # Get basic attributes
  n_traces = f.tracecount
  sample_rate = segyio.tools.dt(f)/1000
  n_samples = f.samples.size
  data = f.trace.raw[:] # get all data into memory
  print('N Traces:', n_traces,' N Samples:', n_samples,' Sample rate:', sample_rate,'ms')

dx = dz = 4;
model_eps1_int4 = data.T[:1500:dz,4500:6000:dx]
x = np.arange(0,model_vp1_int4.shape[0])
z = np.arange(0,model_vp1_int4.shape[1])

plt.figure(figsize=(15,5))
plt.imshow(model_eps1_int4,cmap='jet')
plt.xlabel('Distance (m)')
plt.ylabel('Time (ms)')
plt.title('Epsilon Model Crop')
plt.show()

#%%
filename = '/home/csi-13/Downloads/ModelParams/Delta_Model.sgy'
with segyio.open(filename,ignore_geometry=True) as f:
  # Get basic attributes
  n_traces = f.tracecount
  sample_rate = segyio.tools.dt(f)/1000
  n_samples = f.samples.size
  data = f.trace.raw[:] # get all data into memory
  print('N Traces:', n_traces,' N Samples:', n_samples,' Sample rate:', sample_rate,'ms')

dx = dz = 4;
model_del1_int4 = data.T[:1500:dz,4500:6000:dx]
x = np.arange(0,model_vp1_int4.shape[0])
z = np.arange(0,model_vp1_int4.shape[1])

plt.figure(figsize=(15,5))
plt.imshow(model_del1_int4,cmap='jet')
plt.xlabel('Distance (m)')
plt.ylabel('Time (ms)')
plt.title('Delta Model Crop')
plt.show()

#%% save data

io.savemat('BP_model_crop_int4_375x375.mat',{'model_vp1_int4':model_vp1_int4,
                                 'model_eps1_int4':model_eps1_int4,
                                 'model_del1_int4':model_del1_int4,
                                 'model_thet1_int4':model_thet1_int4,
                                 'x':x1,'z':z1,'dz':dz,'dx':dx})
