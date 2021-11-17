#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 22:17:54 2021

@author: hazwanh
"""

#%% import the module
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy as sp
from scipy import io
import skfmm

from scipy.sparse import csr_matrix, vstack
from scipy.linalg import lstsq, solve
from scipy.sparse.linalg import LinearOperator, cg, lsqr
from scipy.signal import convolve, filtfilt
from scipy import io

from pylops.utils                      import dottest
from pylops.utils.wavelets             import *
from pylops.utils.seismicevents        import *
from pylops.utils.tapers               import *
from pylops.basicoperators             import *
from pylops.signalprocessing           import *
from pylops.waveeqprocessing.lsm import _traveltime_table, Demigration, LSM

from pylops.optimization.leastsquares  import *
from pylops.optimization.sparsity  import *

import math as mt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fseikonal.TTI.ttieikonal as ttieik
import fseikonal.TTI.facttieikonal as fttieik
import time as tm

import math as mt

#%% Set the source and receivers

# datapath = 'path_to_model/model_scattered_100x200.mat' # change datapath to where the model belong
datapath = '/home/hazwanh/Documents/Coding/python/pylops/fast_sweeping/bp_model/model_scattered_100x200.mat'
vel_true = (io.loadmat(datapath)['vp']).T
epsilon_true = (io.loadmat(datapath)['epsilon']).T
delta_true = (io.loadmat(datapath)['delta']).T
theta_true = (io.loadmat(datapath)['theta']).T
x = np.arange(0,vel_true.shape[0])
z = np.arange(0,vel_true.shape[1])

# x = np.arange(0,np.max(x)-np.min(x)+4,4)
# z = np.arange(0,np.max(z)-np.min(z)+4,4)
nx, nz = len(x), len(z)
dx, dz = 4, 4

refl = np.diff(vel_true, axis=1)
refl = np.hstack([refl, np.zeros((nx, 1))])

# Smooth velocity
v0 = 1800 # initial velocity
nsmooth=30
# nsmooth = 30              second test
vel = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel_true, axis=0)
vel = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel, axis=1)

# Receivers
nr = 25
# rx = np.linspace(dx*25, (nx-25)*dx, nr)
rx = np.linspace(dx, nx-dx, nr)
rz = 5*np.ones(nr)
recs = np.vstack((rx, rz))
dr = recs[0,1]-recs[0,0]

# Sources
ns = 25
# sx = np.linspace(dx*25, (nx-25)*dx, ns)
sx = np.linspace(dx, nx-dx, ns)
sz = 5*np.ones(ns)
sources = np.vstack((sx, sz))
ds = sources[0,1]-sources[0,0]

#%% Display the figure

velmin = 1800
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
im = plt.imshow(epsilon_true.T, cmap='jet',
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Epsilon')
plt.ylim(z[-1], z[0])

plt.figure(figsize=(10,5))
im = plt.imshow(delta_true.T, cmap='jet',
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Delta')
plt.ylim(z[-1], z[0])

#%% Generate wavelet and other parameter
nt = 1154
dt = 0.004
t = np.arange(nt)*dt

wav, wavt, wavc = ricker(t[:41], f0=20)

#%% Computes the travel time using eikonal
trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vel, mode='eikonal') 

# Generate the ricker wavelet
itrav_py = (np.floor(trav/dt)).astype(np.int32)
travd_py = (trav/dt - itrav_py)
itrav_py = itrav_py.reshape(nx, nz, ns*nr)
travd_py = travd_py.reshape(nx, nz, ns*nr)

#%% Generate lsm operator, data and madj for pylops fault

Sop_py = Spread(dims=(nx, nz), dimsd=(ns*nr, nt), table=itrav_py, dtable=travd_py, engine='numba')
dottest(Sop_py, ns*nr*nt, nx*nz)
Cop_py = Convolve1D(ns*nr*nt, h=wav, offset=wavc, dims=(ns*nr, nt), dir=1)

LSMop_py = Cop_py*Sop_py
LSMop_py = LinearOperator(LSMop_py, explicit=False)

d_py = LSMop_py * refl.ravel()
d_py = d_py.reshape(ns, nr, nt)

madj_py = LSMop_py.H * d_py.ravel()
madj_py = madj_py.reshape(nx, nz)

#%% Get the inversion 

minv_py_25 = LSMop_py.div(d_py.ravel(), niter=25)
minv_py_25 = minv_py_25.reshape(nx, nz)

minv_py_50 = LSMop_py.div(d_py.ravel(), niter=50)
minv_py_50 = minv_py_50.reshape(nx, nz)

minv_py_100 = LSMop_py.div(d_py.ravel(), niter=100)
minv_py_100 = minv_py_100.reshape(nx, nz)

#%% Display the migrated image

rmin = -np.abs(refl).max()
rmax = np.abs(refl).max()

plt.figure(figsize=(10,5))
im = plt.imshow(refl.T, cmap='gray', vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('true refl')

plt.figure(figsize=(10,5))
im = plt.imshow(madj_py.T, cmap='gray',vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('madj_py')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_py_25.T, cmap='gray',vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('minv_py_25')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_py_50.T, cmap='gray',vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('minv_py_50')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_py_100.T, cmap='gray',vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('minv_py_100')