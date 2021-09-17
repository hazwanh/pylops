#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:32:12 2021

@author: hazwanh
"""

# plt.close('all')
%reset
#%% import the module
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import skfmm

from scipy.sparse import csr_matrix, vstack
from scipy.linalg import lstsq, solve
from scipy.sparse.linalg import LinearOperator, cg, lsqr
from scipy.signal import convolve, filtfilt

from pylops.utils                      import dottest
from pylops.utils.wavelets             import *
from pylops.utils.seismicevents        import *
from pylops.utils.tapers               import *
from pylops.basicoperators             import *
from pylops.signalprocessing           import *
from pylops.waveeqprocessing.lsm_edit import _traveltime_table, Demigration, LSM, zero_offset
from pylops.waveeqprocessing.mdd       import *

from pylops.optimization.leastsquares  import *
from pylops.optimization.sparsity  import *

#%% Create the velocity and reflectivity
# Velocity Model
nx, nz = 301, 100
dx, dz = 4, 4
x, z = np.arange(nx)*dx, np.arange(nz)*dz
v0 = 1000 # initial velocity
kv = 1 # gradient
vel = np.outer(np.ones(nx), v0 +kv*z) 

# Reflectivity Model
refl = np.zeros((nx, nz))
#refl[:, 20] = 1
refl[:, 50] = -1
refl[:, 70] = 0.5

# Receivers
nr = 31
rx = np.linspace(dx*25, (nx-25)*dx, nr)
# rx = np.linspace(dx, (nx)*dx, nr)
rz = 20*np.ones(nr)
recs = np.vstack((rx, rz))
dr = recs[0,1]-recs[0,0]

# Sources
ns = 31
sx = np.linspace(dx*25, (nx-25)*dx, ns)
# sx = np.linspace(dx, (nx)*dx, ns)
sz = 20*np.ones(ns)
sources = np.vstack((sx, sz))
ds = sources[0,1]-sources[0,0]

#%% Display the reflectivity and velocity model
plt.figure(figsize=(10,5))
im = plt.imshow(vel.T, cmap='rainbow', extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('Velocity')
plt.xlim(x[0], x[-1])

plt.figure(figsize=(10,5))
im = plt.imshow(refl.T, cmap='gray', extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('Reflectivity')
plt.xlim(x[0], x[-1]);

#%% Calculate the travel time
trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vel, mode='eikonal')


####
# trav2, trav_srcs2, trav_recs2 = _traveltime_table(z, x, sources, recs, v0, mode='analytic') # ray tracing

#%%
nt = 651
dt = 0.004
t = np.arange(nt)*dt
wav, wavt, wavc = ricker(t[:41], f0=20)

itrav = (np.floor(trav/dt)).astype(np.int32)
travd = (trav/dt - itrav)
itrav = itrav.reshape(nx, nz, ns*nr)
travd = travd.reshape(nx, nz, ns*nr)

# ##### create data and invert the model (marmousi example)
# Sop = Spread(dims=(nx, nz), dimsd=(ns*nr, nt), table=itrav, dtable=travd, engine='numba')
# dottest(Sop, ns*nr*nt, nx*nz)

# Cop = Convolve1D(ns*nr*nt, h=wav, offset=wavc, dims=(ns*nr, nt), dir=1)

# LSMop = Cop*Sop
# LSMop = LinearOperator(LSMop, explicit=False)

# d5 = LSMop * refl.ravel()
# d5 = d5.reshape(ns, nr, nt)

# madj2 = LSMop.H * d5.ravel()
# madj2 = madj2.reshape(nx, nz)
#%%
# create the lsm operator
lsm = LSM(z, x, t, sources, recs, v0, wav, wavc,
          mode='analytic')

# create the seismic data, d:
d = lsm.Demop * refl.ravel()
d = d.reshape(ns, nr, nt)

# a) create the migrated data, mig1:
mig1 = lsm.Demop.H * d.ravel()
mig1 = mig1.reshape(nx, nz)

# # demigrate using lsqr
# d2 = lsqr(lsm.Demop.H, mig1.flatten(), damp=1e-10, iter_lim=20, show=1)[0]
# d2 = d2.reshape(ns, nr, nt)

# inv using hl mdd style
ncp = get_array_module(d)
minv_ls = lsqr(lsm.Demop, d.flatten(), damp=1e-10, iter_lim=20, show=1)[0]
minv_ls = minv_ls.reshape(nx, nz)
wav1 = wav.copy()
wav1 = wav1[ncp.newaxis]
minv_hl = get_fftconvolve(d)(minv_ls, wav1, mode='same')

# remigrate d2:
mig2 = lsm.Demop.H * d2.ravel()
mig2 = mig2.reshape(nx, nz)

# b) demigrate mig1, to obtain data, dmig1:
dmig1 = lsm.Demop * mig1.ravel()
dmig1 = dmig1.reshape(ns, nr, nt)

#%%
# c) re-migrate dmig1:
mig2 = lsm.Demop.H * dmig1.ravel()
mig2 = mig2.reshape(nx, nz)

lsm_H_lsm = lsm.Demop.H * lsm.Demop
mig2_tmp = lsm_H_lsm * mig1.ravel()
mig2_tmp = mig2_tmp.reshape(nx, nz)

dmig2 = lsm.Demop * mig2.ravel()
dmig2 = dmig2.reshape(ns, nr, nt)

# d) create matching filter to match c) to a):


# e) apply the matching filter from d) to the data in a):
rls = lsm_H_lsm.H * mig1.ravel()
rls = rls.reshape(nx, nz)


#%%



#%%

# seismic data, d:
plt.figure(figsize=(10,5))
im = plt.imshow(zero_offset(d).T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('seismic data, d')

# migrated data, mig1:
plt.figure(figsize=(10,5))
im = plt.imshow(mig1.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('migrated data, mig1')

# demigrated data, dmig1:
plt.figure(figsize=(10,5))
im = plt.imshow(zero_offset(dmig1).T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('demigrated data, dmig1')

# remigrated data, mig2:
plt.figure(figsize=(10,5))
im = plt.imshow(mig2.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('migrated data, mig2')

# match filter applied, rls :
plt.figure(figsize=(10,5))
im = plt.imshow(, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('final, rls')

plt.figure(figsize=(10,5))
im = plt.imshow(zero_offset(dmig2).T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('demigrated data, dmig2')

plt.figure(figsize=(10,5))
im = plt.imshow(zero_offset(minv1).T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.title('demigrated data, dmig2')

plt.figure(figsize=(10,5))
im = plt.imshow(refl.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('demigrated data, dmig2')