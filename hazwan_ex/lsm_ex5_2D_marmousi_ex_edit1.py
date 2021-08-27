#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:15:20 2021

@author: csi-13
"""
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
from pylops.waveeqprocessing.lsm_edit import _traveltime_table, Demigration, LSM

from pylops.optimization.leastsquares  import *
from pylops.optimization.sparsity  import *
#%% Generate the marmousi model and display

#Velocity
inputfile='data/avo/poststack_model.npz'

model = np.load(inputfile)
x, z, vel_true = model['x'] - model['x'][0], model['z'] - model['z'][0], 1000*model['model'].T
nx, nz = len(x), len(z)
dx, dz = 4, 4

# Reflectivity
refl = np.diff(vel_true, axis=1)
refl = np.hstack([refl, np.zeros((nx, 1))])

# Smooth velocity
v0 = 1600 # initial velocity
nsmooth=100
vel = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel_true, axis=0)
vel = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel, axis=1)

# Receivers
nr = 31
rx = np.linspace(dx*50, (nx-50)*dx, nr)
rz = 20*np.ones(nr)
recs = np.vstack((rx, rz))
dr = recs[0,1]-recs[0,0]

# Sources
ns = 31
sx = np.linspace(dx*ns, (nx-ns)*dx, ns)
sz = ns*np.ones(ns)
sources = np.vstack((sx, sz))
ds = sources[0,1]-sources[0,0]

#%% Display the figure

plt.figure(figsize=(10,5))
im = plt.imshow(vel_true.T, cmap='rainbow', extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('Velocity')
plt.ylim(z[-1], z[0])

plt.figure(figsize=(10,5))
im = plt.imshow(vel.T, cmap='jet', extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('Smooth velocity')
plt.ylim(z[-1], z[0])

plt.figure(figsize=(10,5))
im = plt.imshow(refl.T, cmap='gray', extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('Reflectivity')
plt.ylim(z[-1], z[0]);
#%% Computes the travel time using eikonal
trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vel, mode='eikonal')   

fig, axs = plt.subplots(1, 3, figsize=(14, 3))
im = axs[0].imshow(trav_srcs[:, ns//2].reshape((nx, nz)).T, cmap='rainbow', 
                   extent = (x[0], x[-1], z[-1], z[0]))
axs[0].scatter(sources[0, ns//2], sources[1, ns//2], marker='*', s=150, c='r', edgecolors='k')
axs[0].axis('tight')
axs[0].set_xlabel('x [m]')
axs[0].set_ylabel('y [m]')
axs[0].set_title('Source traveltime')
axs[0].set_ylim(z[-1], z[0])

plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(trav_recs[:, nr//4].reshape((nx, nz)).T, cmap='rainbow', 
                   extent = (x[0], x[-1], z[-1], z[0]))
axs[1].scatter(recs[0, nr//4], recs[1, nr//4], marker='v', s=150, c='b', edgecolors='k')
axs[1].axis('tight')
axs[1].set_xlabel('x [m]')
axs[1].set_ylabel('y [m]')
axs[1].set_title('Receiver traveltime')
axs[1].set_ylim(z[-1], z[0])

plt.colorbar(im, ax=axs[1])
im = axs[2].imshow(trav[:, ns//2*nr+nr//4].reshape((nx, nz)).T, cmap='rainbow', 
                   extent = (x[0], x[-1], z[-1], z[0]))
axs[2].scatter(sources[0, ns//2], sources[1, ns//2], marker='*', s=150, c='r', edgecolors='k')
axs[2].scatter(recs[0, nr//4], recs[1, nr//4], marker='v', s=150, c='b', edgecolors='k')
axs[2].axis('tight')
axs[2].set_xlabel('x [m]')
axs[2].set_ylabel('y [m]')
axs[2].set_title('Src+rec traveltime')
axs[2].set_ylim(z[-1], z[0])
plt.colorbar(im, ax=axs[2]);

#%% Perform LS on velocity model
nt = 751
dt = 0.004
t = np.arange(nt)*dt

# Generate the ricker wavelet
itrav = (np.floor(trav/dt)).astype(np.int32)
travd = (trav/dt - itrav)
itrav = itrav.reshape(nx, nz, ns*nr)
travd = travd.reshape(nx, nz, ns*nr)

Sop = Spread(dims=(nx, nz), dimsd=(ns*nr, nt), table=itrav, dtable=travd, engine='numba')
dottest(Sop, ns*nr*nt, nx*nz)

wav, wavt, wavc = ricker(t[:41], f0=20)
Cop = Convolve1D(ns*nr*nt, h=wav, offset=wavc, dims=(ns*nr, nt), dir=1)

LSMop = Cop*Sop
LSMop = LinearOperator(LSMop, explicit=False)

d = LSMop * refl.ravel()
d = d.reshape(ns, nr, nt)

madj = LSMop.H * d.ravel()
madj = madj.reshape(nx, nz)

minv = LSMop.div(d.ravel(), niter=100)
minv = minv.reshape(nx, nz)

#%% demigration
dadj = LSMop * madj.ravel()
dadj = dadj.reshape(ns, nr, nt)

dinv = LSMop * minv.ravel()
dinv = dinv.reshape(ns, nr, nt)

#%% Perform LS on reflectivity model
lsm = LSM(z, x, t, sources, recs, vel, wav, wavc,
          mode='eikonal')

# create the data, d:
d = lsm.Demop * refl.ravel()
d = d.reshape(ns, nr, nt)

# Adjoint
madj = lsm.Demop.H * d.ravel()
madj = madj.reshape(nx, nz)

# LS solution
minv = lsm.solve(d.ravel(), solver=lsqr, **dict(iter_lim=100, show=True))
minv = minv.reshape(nx, nz)

# LS solution with sparse model
minv_sparse = lsm.solve(d.ravel(), solver=FISTA, **dict(eps=1e4, niter=100, show=True))
minv_sparse = minv_sparse.reshape(nx, nz)

# SPGL-1
minv_sgpl1 = lsm.solve(d.ravel(), solver=SPGL1, **dict(sigma=1e-5, iter_lim=100, verbosity=2))
minv_sgpl1 = minv_sgpl1.reshape(nx, nz)

#%% Demigration

# adjoint
dadj = LSMop * madj.ravel()
dadj = dadj.reshape(ns, nr, nt)

# LS solution
dinv = LSMop * minv.ravel()
dinv = dinv.reshape(ns, nr, nt)

# LS solution with sparse model
dinv_sparse = LSMop * minv_sparse.ravel()
dinv_sparse = dinv_sparse.reshape(ns, nr, nt)

# SPGL-1
dinv_sgpl1 = LSMop * minv_sgpl1.ravel()
dinv_sgpl1 = dinv_sgpl1.reshape(ns, nr, nt)

#%%
plt.figure(figsize=(10,5))
im = plt.imshow(madj.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'$m_{adj}$')

plt.figure(figsize=(10,5))
im = plt.imshow(minv.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'$m_{inv}$')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_sparse.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'$m_{FISTA}$')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_sgpl1.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'$m_{SPGL-1}$')
###############################################

fig, axs = plt.subplots(1, 3, figsize=(10, 6))
axs[0].imshow(d[ns//2, :, :500].T, cmap='gray')
axs[0].set_title(r'$d$')
axs[0].axis('tight')
axs[1].imshow(dadj[ns//2, :, :500].T, cmap='gray')
axs[1].set_title(r'$d_{adj}$')
axs[1].axis('tight')
axs[2].imshow(dinv[ns//2, :, :500].T, cmap='gray')
axs[2].set_title(r'$d_{inv}$')
axs[2].axis('tight')

fig, axs = plt.subplots(1, 3, figsize=(10, 6))
axs[0].imshow(d[ns//2, :, :500].T, cmap='gray',
              vmin=-0.5*d.max(), vmax=0.5*d.max())
axs[0].set_title(r'$d$')
axs[0].axis('tight')
axs[1].imshow(dadj[ns//2, :, :500].T, cmap='gray',
              vmin=-0.1*dadj.max(), vmax=0.1*dadj.max())
axs[1].set_title(r'$d_{adj}$')
axs[1].axis('tight')
axs[2].imshow(dinv[ns//2, :, :500].T, cmap='gray',
              vmin=-0.5*d.max(), vmax=0.5*d.max())
axs[2].set_title(r'$d_{inv}$')
axs[2].axis('tight');
#%% Display the data and model
fig, axs = plt.subplots(2, 3, figsize=(30, 12))
axs[0][0].imshow(refl.T, cmap='gray')
axs[0][0].axis('tight')
axs[0][0].set_title(r'$m$')
axs[0][1].imshow(madj.T, cmap='gray')
axs[0][1].set_title(r'$m_{adj}$')
axs[0][1].axis('tight')
axs[1][0].imshow(minv.T, cmap='gray')
axs[1][0].axis('tight')
axs[1][0].set_title(r'$m_{inv}$');
axs[1][1].imshow(minv_sparse.T, cmap='gray')
axs[1][1].axis('tight')
axs[1][1].set_title(r'$m_{FISTA}$');
axs[1][2].imshow(minv_sgpl1.T, cmap='gray')
axs[1][2].axis('tight')
axs[1][2].set_title(r'$m_{SPGL1}$');

fig, axs = plt.subplots(1, 5, figsize=(15, 4))
axs[0].imshow(d[ns//2, :, :500].T, cmap='gray')
axs[0].set_title(r'$d$')
axs[0].axis('tight')
axs[1].imshow(dadj[ns//2, :, :500].T, cmap='gray')
axs[1].set_title(r'$d_{adj}$')
axs[1].axis('tight')
axs[2].imshow(dinv[ns//2, :, :500].T, cmap='gray')
axs[2].set_title(r'$d_{inv}$')
axs[2].axis('tight')
axs[3].imshow(dinv_sparse[ns//2, :, :500].T, cmap='gray')
axs[3].set_title(r'$d_{fista}$')
axs[3].axis('tight');
axs[4].imshow(dinv_sgpl1[ns//2, :, :500].T, cmap='gray')
axs[4].set_title(r'$d_{SPGL1}$')
axs[4].axis('tight');

#%% Invert for the velocity model itself
trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vel, mode='eikonal')

itrav = (np.floor(trav/dt)).astype(np.int32)
travd = (trav/dt - itrav)
itrav = itrav.reshape(nx, nz, ns*nr)
travd = travd.reshape(nx, nz, ns*nr)

# Modelling
Sop = Spread(dims=(nx, nz), dimsd=(ns*nr, nt), table=itrav, dtable=travd, engine='numba')
dottest(Sop, ns*nr*nt, nx*nz, verb=True)

# generate the ricker wavelet
wav, wavt, wavc = ricker(t[:41], f0=20)
Cop = Convolve1D(ns*nr*nt, h=wav, offset=wavc, dims=(ns*nr, nt), dir=1)
Dop = FirstDerivative(nx*nz, dims=(nx, nz), dir=1)
LSMop = Cop*Sop*Dop
LSMop = LinearOperator(LSMop, explicit=False)

d = LSMop * np.log(vel_true).ravel()
d = d.reshape(ns, nr, nt)
dback = LSMop * np.log(vel).ravel()
dback = dback.reshape(ns, nr, nt)

madj = (Cop*Sop).H * d.ravel()
madj = madj.reshape(nx, nz)

# Regularized inversion
Regop = SecondDerivative(nx*nz, (nx, nz), dir=0)
minv = RegularizedInversion(LSMop, [Regop], d.ravel(),
                            x0=np.log(vel).ravel(),
                            epsRs=[1e1], returninfo=False,
                            **dict(iter_lim=200, show=True))
minv = np.exp(minv.reshape((nx, nz)))

# demigration
dadj = (Cop*Sop) * madj.ravel()
dadj = dadj.reshape(ns, nr, nt)

dinv = LSMop *  np.log(minv).ravel()
dinv = dinv.reshape(ns, nr, nt)

#%% Display the data and model
v0 = 1600
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs[0][0].imshow(vel_true.T, cmap='rainbow', vmin=v0, vmax=vel.max())
axs[0][0].axis('tight')
axs[0][0].set_title(r'$m$')
axs[0][1].imshow(vel.T, cmap='rainbow', vmin=vel_true.min(), vmax=vel_true.max())
axs[0][1].set_title(r'$m_{back}$')
axs[0][1].axis('tight')
axs[1][0].imshow(madj.T, cmap='gray')
axs[1][0].axis('tight')
axs[1][0].set_title(r'$m_{adj}$');
axs[1][1].imshow(minv.T, cmap='rainbow', vmin=vel_true.min(), vmax=vel_true.max())
axs[1][1].axis('tight')
axs[1][1].set_title(r'$m_{inv}$');

fig, axs = plt.subplots(1, 3, figsize=(14, 8))
axs[0].imshow(d[ns//2, :, :500].T, cmap='gray',
              vmin=-0.5*d.max(), vmax=0.5*d.max())
axs[0].set_title(r'$d$')
axs[0].axis('tight')
axs[1].imshow(dadj[ns//2, :, :500].T, cmap='gray',
              vmin=-0.1*dadj.max(), vmax=0.1*dadj.max())
axs[1].set_title(r'$d_{adj}$')
axs[1].axis('tight')
axs[2].imshow(dinv[ns//2, :, :500].T, cmap='gray',
              vmin=-0.5*d.max(), vmax=0.5*d.max())
axs[2].set_title(r'$d_{inv}$')
axs[2].axis('tight');

plt.figure(figsize=(3, 9))
plt.plot(vel_true[100], z, 'k')
plt.plot(vel[100], z, '--k')
plt.plot(minv[100], z, 'r')
plt.gca().invert_yaxis()