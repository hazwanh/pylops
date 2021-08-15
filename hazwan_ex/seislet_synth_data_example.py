#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:32:48 2021

@author: hazwanh
"""
#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pylops

from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from pylops.utils                      import dottest
from pylops.utils.wavelets             import *
from pylops.utils.seismicevents        import *
from pylops.utils.signalprocessing     import slope_estimate
from pylops.basicoperators             import *
from pylops.signalprocessing           import *
from pylops.avo.poststack              import *

from pylops.signalprocessing.Seislet import _predict_trace

#%% Data creation
np.random.seed(0)

# reflectivity model
nx, nt = 2**7, 121
dx, dt = 4, 0.004
x, t = np.arange(nx)*dx, np.arange(nt)*dt

layers = np.cumsum(np.random.uniform(2, 20, 40).astype(np.int))
layers =layers[layers<nt]
nlayers = len(layers)

k = np.tan(np.deg2rad(45))
ai = 1500 * np.ones(nt)
for top, base in zip(layers[:-1], layers[1:]):
    ai[top:base] = np.random.normal(2000, 200)
    
refl = np.pad(np.diff(ai), (0, 1))
refl[-30:] = 0

# wavelet
ntwav = 41
f0 = 30
wav = pylops.utils.wavelets.ricker(np.arange(ntwav)*0.004, f0)[0]
wavc = np.argmax(wav)

# trace
trace = np.convolve(refl, wav, mode='same')

theta = np.linspace(10, 10, nx)
slope = np.outer(np.ones(nt), np.deg2rad(theta) * dt / dx)
d = np.zeros((nt, nx))
tr = trace.copy()
for ix in range(nx):
    tr = _predict_trace(tr, t, dt, dx, slope[:, ix])
    d[:, ix] = tr
plt.figure()
plt.imshow(d, cmap='gray', vmin=-200, vmax=200);

#%% slope estimation

slope_est = -slope_estimate(d, dt, dx, smooth=7)[0]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(d, cmap='gray', vmin=-200, vmax=200,
              extent = (x[0], x[-1], t[-1], t[0]))
axs[0].set_title('Data')
axs[0].axis('tight')
axs[1].imshow(slope, cmap='seismic', vmin=0, vmax=slope.max(),
              extent = (x[0], x[-1], t[-1], t[0]))
axs[1].set_title('Slopes')
axs[1].axis('tight')
im = axs[2].imshow(slope_est, cmap='seismic', vmin=0, vmax=slope.max(),
                   extent = (x[0], x[-1], t[-1], t[0]))
axs[2].set_title('Est Slopes')
axs[2].axis('tight')
plt.colorbar(im, ax=axs[2]);

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
im = ax.imshow(np.rad2deg(slope_est)*dx/dt, cmap='seismic', vmin=0, vmax=20,
              extent = (x[0], x[-1], t[-1], t[0]))
ax.set_title('Est Angles')
ax.axis('tight')
plt.colorbar(im, ax=ax);

#%% Seislet transform

# Check the dottest
Sop = Seislet(slope.T, sampling=(dx, dt), level=None)
S1op = Seislet(slope_est.T, sampling=(dx, dt), level=None)

dottest(Sop, nt*nx, nt*nx, verb=True);

#%% Apply the transform

seis = Sop * d.T.ravel()
drec = Sop.inverse(seis)

seis1 = S1op * d.T.ravel()
drec1 = S1op.inverse(seis1)

seis = seis.reshape(nx, nt).T
seis1 = seis1.reshape(nx, nt).T
drec = drec.reshape(nx, nt).T
drec1 = drec1.reshape(nx, nt).T

#%% display

fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(d, cmap='gray', vmin=-200, vmax=200)
axs[0].set_title('Data')
axs[1].imshow(seis, cmap='gray', vmin=-200, vmax=200)
axs[1].set_title('Seislet slope')
axs[2].imshow(drec, cmap='gray', vmin=-200, vmax=200)
axs[2].set_title('Inverse')
axs[3].imshow(d-drec, cmap='gray', vmin=-200, vmax=200)
axs[3].set_title('Error');

fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(d, cmap='gray', vmin=-200, vmax=200)
axs[0].set_title('Data')
axs[1].imshow(seis1, cmap='gray', vmin=-200, vmax=200)
axs[1].set_title('Seislet slope_est')
axs[2].imshow(drec1, cmap='gray', vmin=-200, vmax=200)
axs[2].set_title('Inverse')
axs[3].imshow(d-drec1, cmap='gray', vmin=-200, vmax=200)
axs[3].set_title('Error');

nlevels_max = int(np.log2(nx))
levels_size = np.flip(np.array([2 ** i for i in range(nlevels_max)]))
levels_cum = np.cumsum(levels_size)

plt.figure(figsize=(15, 5))
plt.imshow(seis, cmap='gray', vmin=-100, vmax=100)
for level in levels_cum:
    plt.axvline(level-0.5, color='w')
plt.axis('tight');