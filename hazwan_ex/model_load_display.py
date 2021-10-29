#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 17:32:22 2021

@author: hazwanh
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy as sp
from scipy import io
from scipy.signal import convolve, filtfilt

#%%

datapath = '/home/hazwanh/Documents/pylops/fast_sweeping/bp_model/bpmodel_salt_375x750_3750x6750.mat'
# datapath = '/home/hazwanh/Documents/Coding/python/pylops/fast_sweeping/bp_model/bpmodel_salt_375x750_3750x6750.mat'
vel_true = (io.loadmat(datapath)['model_vp1_int4']).T
epsilon_true = (io.loadmat(datapath)['model_eps1_int4']).T
delta_true = (io.loadmat(datapath)['model_del1_int4']).T
theta_true = (io.loadmat(datapath)['model_thet1_int4']).T
x = io.loadmat(datapath)['x']
z = io.loadmat(datapath)['z']

x = np.arange(0,np.max(x)-np.min(x)+4,4)
z = np.arange(0,np.max(z)-np.min(z)+4,4)
nx, nz = len(x), len(z)
dx, dz = 4, 4

refl = np.diff(vel_true, axis=1)
refl = np.hstack([refl, np.zeros((nx, 1))])

# Smooth velocity
v0 = 1492 # initial velocity
nsmooth=30
vel = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel_true, axis=0)
vel = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel, axis=1)
epsilon = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, epsilon_true, axis=0)
epsilon = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, epsilon, axis=1)
delta = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, delta_true, axis=0)
delta = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, delta, axis=1)
theta = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, theta_true, axis=0)
theta = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, theta, axis=1)

# Receivers
nr = 61
rx = np.linspace(dx*25, (nx-25)*dx, nr)
# rx = np.linspace(dx, (nx)*dx, nr)
rz = 20*np.ones(nr)
recs = np.vstack((rx, rz))
dr = recs[0,1]-recs[0,0]

# Sources
ns = 61
sx = np.linspace(dx*25, (nx-25)*dx, ns)
# sx = np.linspace(dx, (nx)*dx, ns)
sz = 20*np.ones(ns)
sources = np.vstack((sx, sz))
ds = sources[0,1]-sources[0,0]

#%% Generate image for each model

velmin = 1492
velmax = np.abs(-1*vel_true).max()

plt.figure(figsize=(10,5))
im = plt.imshow(vel_true.T, cmap='jet', vmin = velmin, vmax = velmax,
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Velocity')
plt.ylim(z[-1], z[0])

plt.figure(figsize=(10,5))
im = plt.imshow(vel.T, cmap='jet', vmin = velmin, vmax = velmax, 
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Smooth velocity')
plt.ylim(z[-1], z[0])

reflmax = np.abs(-1*refl).max()
reflmin = -np.abs(-1*refl).max()

plt.figure(figsize=(10,5))
im = plt.imshow(refl.T, cmap='gray', vmin = reflmin, vmax = reflmax, 
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Reflectivity')
plt.ylim(z[-1], z[0]);

plt.figure(figsize=(10,5))
im = plt.imshow(epsilon.T, cmap='jet',
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Epsilon')
plt.ylim(z[-1], z[0])

plt.figure(figsize=(10,5))
im = plt.imshow(delta.T, cmap='jet',
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Delta')
plt.ylim(z[-1], z[0])