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
inputfile='../pylops_notebooks/data/avo/poststack_model.npz'

model = np.load(inputfile)
x, z, vel_true = model['x'][625] - model['x'][250], model['z'] - model['z'][0], 1000*model['model'].T
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
nr = 15
rx = np.linspace(dx*50, (nx-50)*dx, nr)
rz = 20*np.ones(nr)
recs = np.vstack((rx, rz))
dr = recs[0,1]-recs[0,0]

# Sources
ns = 15
sx = np.linspace(dx*50, (nx-50)*dx, ns)
sz = 20*np.ones(ns)
sources = np.vstack((sx, sz))
ds = sources[0,1]-sources[0,0]

#Set the tom data:
refl_tom = refl[:,188:376]
z_tom = z[188:376]
nz_tom = len(z_tom)

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

#TOM
plt.figure(figsize=(10,5))
im = plt.imshow(refl_tom.T, cmap='gray', extent = (x[0], x[-1], z_tom[-1], z_tom[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('Reflectivity')
plt.ylim(z_tom[-1], z_tom[0]);
#%% Computes the travel time using eikonal
trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vel, mode='eikonal')   

#TOM
trav_tom, trav_srcs_tom, trav_recs_tom = _traveltime_table(z_tom, x, sources, recs, vel[:,188:376], mode='eikonal')  

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

#%% tom
itrav_tom = (np.floor(trav_tom/dt)).astype(np.int32)
travd_tom = (trav_tom/dt - itrav_tom)
itrav_tom = itrav_tom.reshape(nx, nz_tom, ns*nr)
travd_tom = travd_tom.reshape(nx, nz_tom, ns*nr)

Sop_tom = Spread(dims=(nx, nz_tom), dimsd=(ns*nr, nt), table=itrav_tom, dtable=travd_tom, engine='numba')
dottest(Sop_tom, ns*nr*nt, nx*nz_tom)

LSMop_tom = Cop*Sop_tom
LSMop_tom = LinearOperator(LSMop_tom, explicit=False)

d_tom = LSMop_tom * refl_tom.ravel()
d_tom = d_tom.reshape(ns, nr, nt)

madj_tom = LSMop_tom.H * d_tom.ravel()
madj_tom = madj_tom.reshape(nx, nz_tom)

minv_tom = LSMop_tom.div(d_tom.ravel(), niter=100)
minv_tom = minv_tom.reshape(nx, nz_tom)
#####
d_tom_vel = LSMop_tom * refl_tom.ravel()
d_tom_vel = d_tom_vel.reshape(ns, nr, nt)

madj_tom_vel = LSMop_tom.H * d_tom_vel.ravel()
madj_tom_vel = madj_tom_vel.reshape(nx, nz_tom)

minv_tom_vel = LSMop_tom.div(d_tom_vel.ravel(), niter=100)
minv_tom_vel = minv_tom_vel.reshape(nx, nz_tom)
#%% demigration
dadj = LSMop * madj.ravel()
dadj = dadj.reshape(ns, nr, nt)

dinv = LSMop * minv.ravel()
dinv = dinv.reshape(ns, nr, nt)

#%% demigration tom
dadj_tom = LSMop_tom * madj_tom.ravel()
dadj_tom = dadj_tom.reshape(ns, nr, nt)

dinv_tom = LSMop_tom * minv_tom.ravel()
dinv_tom = dinv_tom.reshape(ns, nr, nt)

#%% Perform LS on reflectivity model
lsm = LSM(z, x, t, sources, recs, vel, wav, wavc,
          mode='eikonal')
d = lsm.Demop * refl.ravel()
d = d.reshape(ns, nr, nt)

#tom
lsm_tom = LSM(z_tom, x, t, sources, recs, vel[:,188:376], wav, wavc,
          mode='eikonal')
d_tom = lsm_tom.Demop * refl_tom.ravel()
d_tom = d_tom.reshape(ns, nr, nt)

# Adjoint
madj = lsm.Demop.H * d.ravel()
madj = madj.reshape(nx, nz)
# tom
madj_tom = lsm_tom.Demop.H * d_tom.ravel()
madj_tom = madj_tom.reshape(nx, nz_tom)

# LS solution
minv = lsm.solve(d.ravel(), solver=lsqr, **dict(iter_lim=100, show=True))
minv = minv.reshape(nx, nz)
# tom
minv_tom = lsm_tom.solve(d_tom.ravel(), solver=lsqr, **dict(iter_lim=100, show=True))
minv_tom = minv_tom.reshape(nx, nz_tom)

# LS solution with sparse model
minv_sparse = lsm.solve(d.ravel(), solver=FISTA, **dict(eps=1e4, niter=100, show=True))
minv_sparse = minv_sparse.reshape(nx, nz)
# tom
minv_sparse_tom = lsm_tom.solve(d_tom.ravel(), solver=FISTA, **dict(eps=1e4, niter=100, show=True))
minv_sparse_tom = minv_sparse_tom.reshape(nx, nz_tom)

#%% Demigration

# adjoint
dadj = LSMop * madj.ravel()
dadj = dadj.reshape(ns, nr, nt)
# tom
dadj_tom = LSMop_tom * madj_tom.ravel()
dadj_tom = dadj_tom.reshape(ns, nr, nt)

# LS solution
dinv = LSMop * minv.ravel()
dinv = dinv.reshape(ns, nr, nt)
# tom
dinv_tom = LSMop_tom * minv_tom.ravel()
dinv_tom = dinv_tom.reshape(ns, nr, nt)

# LS solution with sparse model
dinv_sparse = LSMop * minv_sparse.ravel()
dinv_sparse = dinv_sparse.reshape(ns, nr, nt)
# tom
dinv_sparse_tom = LSMop_tom * minv_sparse_tom.ravel()
dinv_sparse_tom = dinv_sparse_tom.reshape(ns, nr, nt)

#%%
plt.figure(figsize=(10,5))
im = plt.imshow(madj.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'Full $m_{adj}$')

plt.figure(figsize=(10,5))
im = plt.imshow(minv.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'Full $m_{inv}$')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_sparse.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'Full $m_{FISTA}$')

#####
plt.figure(figsize=(10,5))
im = plt.imshow(madj[:,188:376].T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'Cropped Full $m_{adj}$')

plt.figure(figsize=(10,5))
im = plt.imshow(minv[:,188:376].T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'Cropped Full $m_{inv}$')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_sparse[:,188:376].T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'Cropped Full $m_{FISTA}$')

plt.figure(figsize=(10,5))
im = plt.imshow(madj.T, cmap='gray', extent = [x.min(),x.max(),
                                  t.max(),t.min()], aspect='auto')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('Offset [m]'),plt.ylabel('Time [s]')
plt.title(r'Full $m_{adj}$')

plt.figure(figsize=(10,5))
im = plt.imshow(minv.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'Full $m_{inv}$')

#%%
plt.figure(figsize=(10,5))
im = plt.imshow(madj_tom.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'Target oriented $m_{adj}$')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_tom.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'Target oriented $m_{inv}$')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_sparse_tom.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title(r'Target oriented $m_{FISTA}$')

#tom
plt.figure(figsize=(10,5))
im = plt.imshow(minv_sparse_tom.T, cmap='gray', extent = (x[0], x[-1], z_tom[-1], z_tom[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('Reflectivity')
plt.ylim(z_tom[-1], z_tom[0]);

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