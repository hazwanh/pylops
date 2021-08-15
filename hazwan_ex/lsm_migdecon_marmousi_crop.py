#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:15:20 2021

@author: csi-13
"""
plt.close('all')
%reset
#%% import the module
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from numpy.linalg import norm
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
inputfile='../pylops_notebooks/data/avo/poststack_model.npz'

model = np.load(inputfile)
x, z, vel_true = (model['x'] - model['x'][0])[250:625], (model['z'] - model['z'][0])[0:375], (1000*model['model'].T)[250:625,0:375]
x = np.arange(0,max(x)-min(x)+1, 4)
nx, nz = len(x), len(z)
dx, dz = 4, 4

# Reflectivity
refl = np.diff(vel_true, axis=1)
refl = np.hstack([refl, np.zeros((nx, 1))])

# Smooth velocity
v0 = 1600 # initial velocity
nsmooth=30
vel = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel_true, axis=0)
vel = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel, axis=1)

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

#%% Display the figure

velmin = 1600
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

#%% Computes the travel time using eikonal
trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vel, mode='eikonal')   
#%% Perform LS on velocity model
nt = 400
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
d = lsm.Demop * refl.ravel()
d = d.reshape(ns, nr, nt)

# a) create the migrated data, mig1:
mig1 = lsm.Demop.H * d.ravel()
mig1 = mig1.reshape(nx, nz)

# b) demigrate mig1, to obtain data, dmig1:
dmig1 = lsm.Demop * mig1.ravel()
dmig1 = dmig1.reshape(ns, nr, nt)

# LS solution
# minv = lsm.solve(d.ravel(), solver=lsqr, **dict(iter_lim=100, show=True))
# minv = minv.reshape(nx, nz)

#%% Demigration

# adjoint
dadj = LSMop * madj.ravel()
dadj = dadj.reshape(ns, nr, nt)

# LS solution
dinv = LSMop * minv.ravel()
dinv = dinv.reshape(ns, nr, nt)

#%%
plt.figure(figsize=(10,5))
im = plt.imshow(madj.T, cmap='gray', extent = [x.min(),x.max(),
                                  t.max(),t.min()], aspect='auto')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('time [s]')
plt.title(r'Full $m_{adj}$')

plt.figure(figsize=(10,5))
im = plt.imshow(minv.T, cmap='gray', extent = [x.min(),x.max(),
                                  t.max(),t.min()], aspect='auto')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('time [s]')
plt.title(r'Full $m_{inv}$')

#%% Display the data and model
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
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

fig, axs = plt.subplots(1, 5, figsize=(10, 4))
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

#%%
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

#%% demigration
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

#%%
def zero_offset(data):
    
    nt = data.shape[2]
    nx = data.shape[1]
    
    zo = np.zeros([nx,nt])
    
    for i in range(nx):
        zo[i,:] = data[i,i,:]
        
    return zo 