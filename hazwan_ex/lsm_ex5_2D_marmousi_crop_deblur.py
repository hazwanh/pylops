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
lsm = LSM(z, x, t, sources, recs, vel, wav, wavc, mode='eikonal')
d = lsm.Demop * refl.ravel()
d = d.reshape(ns, nr, nt)

# Adjoint
madj = lsm.Demop.H * d.ravel()
madj = madj.reshape(nx, nz)

# LS solution
minv = lsm.solve(d.ravel(), solver=lsqr, **dict(iter_lim=10, show=True))
minv = minv.reshape(nx, nz)

# create the LhL operator
lsmhl_op = lsm.Demop.H*lsm.Demop
madj2 = lhlop.H * madj.ravel()
madj2 = madj2.reshape(nx, nz)

# fista solution
minv_sparse = lsm.solve(d.ravel(), solver=FISTA, **dict(eps=1e4, niter=30, show=True))
minv_sparse = minv_sparse.reshape(nx, nz)

# Create the 2D Transform wavelet operator
Nz, Nx = refl.shape
Wop = DWT2D((Nz, Nx), wavelet='haar', level=3)
Dop = [FirstDerivative(Nz * Nx, dims=(Nz, Nx), dir=0, edge=False),
       FirstDerivative(Nz * Nx, dims=(Nz, Nx), dir=1, edge=False)]
DWop = Dop + [Wop, ]

# deblurring using NormalEquationInversion without regularization
minv_imdb = NormalEquationsInversion(lsm.Demop,None,d.ravel(),maxiter=10)
minv_imdb = minv_imdb.reshape(nx, nz)

minv_imdb_hl = NormalEquationsInversion(lsmhl_op,None,madj.ravel(),maxiter=10)
minv_imdb_hl = minv_imdb_hl.reshape(nx, nz)

minv_imdbr = NormalEquationsInversion(lsm.Demop,Dop,d.ravel(),maxiter=10)
minv_imdbr = minv_imdbr.reshape(nx, nz) # no effect

minv_imdbw = NormalEquationsInversion(lsm.Demop * Wop.H,None,d.ravel(),maxiter=10)
minv_imdbw = Wop.H * minv_imdb2
minv_imdbw= minv_imdbw.reshape(nx, nz) # no effect

dinv_imdbhl = LSMop * minv_imdb_hl.ravel()
dinv_imdbhl = dinv_imdbhl.reshape(ns, nr, nt)
minv_imdbhlsqr = lsm.solve(dinv_imdbhl.ravel(), solver=FISTA, **dict(eps=1e4, niter=30, show=True))
minv_imdbhlsqr = minv_imdbhlsqr.reshape(nx, nz)

# deblurring using split-bregman
Dop = [FirstDerivative(Nz * Nx, dims=(Nz, Nx), dir=0, edge=False),
       FirstDerivative(Nz * Nx, dims=(Nz, Nx), dir=1, edge=False)]
minv_sb = SplitBregman(lsm.Demop, Dop, d.flatten(), niter_outer=5, niter_inner=3,
                       mu=1.5, epsRL1s=[1e0, 1e0],tol=1e-4, tau=1., show=False,
                       ** dict(iter_lim=5, damp=1e-4))[0]
minv_sb = minv_sb.reshape(nx, nz)

minv_sb_hl = SplitBregman(lsmhl_op, Dop, madj.flatten(), niter_outer=5, niter_inner=3,
                       mu=1.5, epsRL1s=[1e0, 1e0],tol=1e-4, tau=1., show=False,
                       ** dict(iter_lim=5, damp=1e-4))[0]
minv_sb_hl = minv_sb_hl.reshape(nx, nz)

dinv_sbhl = lsm.Demop * minv_sb_hl.ravel()
dinv_sbhl = dinv_sbhl.reshape(ns, nr, nt)
minv_sbhlsqr = lsm.solve(dinv_sbhl.ravel(), solver=FISTA, **dict(eps=1e4, niter=60, show=True))
minv_sbhlsqr = minv_sbhlsqr.reshape(nx, nz)

# # sb with fftconvolve
# ncp = get_array_module(d)
# wav1 = wav.copy()
# wav1 = wav1[ncp.newaxis]
# minv_hlsb = get_fftconvolve(d)(minv_sb.flatten(), wav1, mode='same')
# minv_hlsb = minv_hlsb.reshape(nx, nz)

# Apply deblurring using FISTA
minv_dbf = FISTA(lsm.Demop * Wop.H, d.flatten(), eps=1e-1, niter=10)[0]
minv_dbf = Wop.H * minv_dbf
minv_dbf = minv_dbf.reshape(nx, nz)

minv_dbf_hl = FISTA(lsmhl_op * Wop.H, madj.flatten(), eps=1e-1, niter=10)[0]
minv_dbf_hl = Wop.H * minv_dbf_hl
minv_dbf_hl = minv_dbf_hl.reshape(nx, nz)

#%% Demigration

# adjoint
dadj = LSMop * madj.ravel()
dadj = dadj.reshape(ns, nr, nt)

# LS solution
dinv = LSMop * minv.ravel()
dinv = dinv.reshape(ns, nr, nt)

#%%
rmin = -np.abs(refl).max()
rmax = np.abs(refl).max()

# true refl
plt.figure(figsize=(10,5))
im = plt.imshow(refl.T, cmap='gray', vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('true refl')

# madj
plt.figure(figsize=(10,5))
im = plt.imshow(madj.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('madj')

# minv_ls
plt.figure(figsize=(10,5))
im = plt.imshow(minv.T, cmap='gray', vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('minv')

# minv NormalEquationInversion
plt.figure(figsize=(10,5))
im = plt.imshow(minv_imdb.T, cmap='gray', vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('minv_imdb')

# minv split bergman
plt.figure(figsize=(10,5))
im = plt.imshow(minv_sb.T, cmap='gray', vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('minv_sb')

# minv deblur fista with regularization
plt.figure(figsize=(10,5))
im = plt.imshow(minv_dbf.T, cmap='gray', vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('minv_dbf')
#%%
def zero_offset(data):
    
    nt = data.shape[2]
    nx = data.shape[1]
    
    zo = np.zeros([nx,nt])
    
    for i in range(nx):
        zo[i,:] = data[i,i,:]
        
    return zo 