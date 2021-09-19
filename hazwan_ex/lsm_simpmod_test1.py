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

#%%
plt.figure(figsize=(10,5))
im = plt.imshow(trav.T, cmap='jet')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('Total traveltime')

plt.figure(figsize=(10,5))
im = plt.imshow(trav_srcs.T, cmap='jet')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('Source-to-subsurface traveltime')

plt.figure(figsize=(10,5))
im = plt.imshow(trav_recs.T, cmap='jet')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('Receiver-to-subsurface traveltime')

#%% Create the wavelet
nt = 651
dt = 0.004
t = np.arange(nt)*dt
wav, wavt, wavc = ricker(t[:41], f0=20)

itrav = (np.floor(trav/dt)).astype(np.int32)
travd = (trav/dt - itrav)
itrav = itrav.reshape(nx, nz, ns*nr)
travd = travd.reshape(nx, nz, ns*nr)

"""
# Perform LS on velocity model

Sop = Spread(dims=(nx, nz), dimsd=(ns*nr, nt), table=itrav, dtable=travd, engine='numba')
dottest(Sop, ns*nr*nt, nx*nz)

wav, wavt, wavc = ricker(t[:41], f0=20)
Cop = Convolve1D(ns*nr*nt, h=wav, offset=wavc, dims=(ns*nr, nt), dir=1)

LSMop = Cop*Sop
LSMop = LinearOperator(LSMop, explicit=False)

# create the data
d_vel = LSMop * refl.ravel()
d_vel = d_vel.reshape(ns, nr, nt)

# migration
mig_vel = LSMop.H * d_vel.ravel()
mig_vel = mig_vel.reshape(nx, nz)

minv_vel = LSMop.div(d_vel.ravel(), niter=100)
minv_vel = minv_vel.reshape(nx, nz)

# demigration
dadj_vel = LSMop * mig_vel.ravel()
dadj_vel = dadj_vel.reshape(ns, nr, nt)

dinv_vel = LSMop * minv_vel.ravel()
dinv_vel = dinv_vel.reshape(ns, nr, nt)
"""
"""
plt.figure(figsize=(10,5))
im = plt.imshow(mig_vel.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('migrated velocity data, mig_vel')
"""

#%% Create the operator

# create the lsm operator
lsm = LSM(z, x, t, sources, recs, v0, wav, wavc,
          mode='analytic')

# create the seismic data, d:
d = lsm.Demop * refl.ravel()
d = d.reshape(ns, nr, nt)

# a) create the migrated data, mig1:
mig1 = lsm.Demop.H * d.ravel()
mig1 = mig1.reshape(nx, nz)

# default
minv = lsm.solve(d.ravel(), solver=lsqr, **dict(iter_lim=10))
minv = minv.reshape(nx, nz)

#%% Apply deblurring
"""
Cop = pylops.signalprocessing.Convolve2D(Nz * Nx, h=h,
                                         offset=(nh[0] // 2,
                                                 nh[1] // 2),
                                         dims=(Nz, Nx), dtype='float32')

Wop = pylops.signalprocessing.DWT2D((Nz, Nx), wavelet='haar', level=3)

imdeblurfista = \
    pylops.optimization.sparsity.FISTA(Cop * Wop.H, imblur, eps=1e-1,
                                       niter=100)[0]
imdeblurfista = Wop.H * imdeblurfista
"""
# Get the size
Nz, Nx = mig1.shape

# using deblurr (NormalEquationInversion)
minv_imdb = NormalEquationsInversion(lsm.Demop,None,d.ravel(),maxiter=10)
minv_imdb = minv_imdb.reshape(nx, nz)

# lsqr (similar to std lsm)
minv_lsqr = lsqr(lsm.Demop, d.flatten(), damp=1e-10, iter_lim=10, show=1)[0]
minv_lsqr = minv_lsqr.reshape(nx, nz)

# using split-bergman
Dop = [FirstDerivative(Nz * Nx, dims=(Nz, Nx), dir=0, edge=False),
       FirstDerivative(Nz * Nx, dims=(Nz, Nx), dir=1, edge=False)]
minv_sb = SplitBregman(lsm.Demop, Dop, d.flatten(), niter_outer=10, niter_inner=5,
                       mu=1.5, epsRL1s=[1e0, 1e0],tol=1e-4, tau=1., show=False,
                       ** dict(iter_lim=5, damp=1e-4))[0]
minv_sb = minv_sb.reshape(nx, nz)

# sb with fftconvolve
ncp = get_array_module(d)
wav1 = wav.copy()
wav1 = wav1[ncp.newaxis]
minv_hlsb = get_fftconvolve(d)(minv_sb.flatten(), wav1, mode='same')
minv_hlsb = minv_hlsb.reshape(nx, nz)

# Create the 2D Transform wavelet operator
Wop = DWT2D((Nz, Nx), wavelet='haar', level=3)
Dop = [FirstDerivative(Nz * Nx, dims=(Nz, Nx), dir=0, edge=False),
       FirstDerivative(Nz * Nx, dims=(Nz, Nx), dir=1, edge=False)]
DWop = Dop + [Wop, ]

# Apply deblurring using FISTA
minv_dbf = FISTA(lsm.Demop * Wop.H, d.flatten(), eps=1e-1, niter=10)[0]
minv_dbf = Wop.H * minv_dbf
minv_dbf = minv_dbf.reshape(nx, nz)

# Apply deblurring using sb + haar
minv_tv1 = SplitBregman(lsm.Demop, DWop,d.flatten(),niter_outer=10, niter_inner=5,
                        mu=1.5, epsRL1s=[1e0, 1e0, 1e0], tol=1e-4, tau=1., show=False,
                        ** dict(iter_lim=5, damp=1e-4))[0]
minv_tv1 = minv_tv1.reshape(nx, nz)

#%% Display

# true reflectivity
plt.figure(figsize=(10,5))
im = plt.imshow(refl.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('true reflectivity')

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

#%% invert for the model again (update velocity)

trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vel, mode='eikonal')

itrav = (np.floor(trav/dt)).astype(np.int32)
travd = (trav/dt - itrav)
itrav = itrav.reshape(nx, nz, ns*nr)
travd = travd.reshape(nx, nz, ns*nr)

# Modelling
Sop = Spread(dims=(nx, nz), dimsd=(ns*nr, nt), table=itrav, dtable=travd, engine='numba')
dottest(Sop, ns*nr*nt, nx*nz, verb=True)

wav, wavt, wavc = ricker(t[:41], f0=20)
Cop = Convolve1D(ns*nr*nt, h=wav, offset=wavc, dims=(ns*nr, nt), dir=1)
Dop = FirstDerivative(nx*nz, dims=(nx, nz), dir=1)
LSMop = Cop*Sop*Dop
LSMop = LinearOperator(LSMop, explicit=False)

d_2 = LSMop * np.log(vel).ravel()
d_2 = d_2.reshape(ns, nr, nt)
dback = LSMop * np.log(vel).ravel()
dback = dback.reshape(ns, nr, nt)

madj_2 = (Cop*Sop).H * d_2.ravel()
madj_2 = madj_2.reshape(nx, nz)

# Regularized inversion
Regop = SecondDerivative(nx*nz, (nx, nz), dir=0)
minv_2 = RegularizedInversion(LSMop, [Regop], d_2.ravel(),
                            x0=np.log(vel).ravel(),
                            epsRs=[1e1], returninfo=False,
                            **dict(iter_lim=200, show=True))
minv_2 = np.exp(minv_2.reshape((nx, nz)))

# demigration
dadj_2 = (Cop*Sop) * madj_2.ravel()
dadj_2 = dadj_2.reshape(ns, nr, nt)

dinv_2 = LSMop *  np.log(minv_2).ravel()
dinv_2 = dinv_2.reshape(ns, nr, nt)

# velocity inverted data, minv_2:
plt.figure(figsize=(10,5))
im = plt.imshow(minv_2.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('inverted velocity data, minv_2')

# migrated data, madj_2:
plt.figure(figsize=(10,5))
im = plt.imshow(madj_2.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('migrated data, madj_2')

plt.figure(figsize=(10,5))
im = plt.imshow(zero_offset(d_2).T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('seismic data, d_2')