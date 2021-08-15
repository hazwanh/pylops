#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 16:35:09 2021

@author: hazwanh
"""

# plt.close('all')
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

#%%
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

#%%
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

#%%
trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vel, mode='eikonal')

#%%
nt = 651
dt = 0.004
t = np.arange(nt)*dt
wav, wavt, wavc = ricker(t[:41], f0=20)

itrav = (np.floor(trav/dt)).astype(np.int32)
travd = (trav/dt - itrav)
itrav = itrav.reshape(nx, nz, ns*nr)
travd = travd.reshape(nx, nz, ns*nr)

#%%
lsm = LSM(z, x, t, sources, recs, v0, wav, wavc,
          mode='analytic')

d = lsm.Demop * refl.ravel()
d = d.reshape(ns, nr, nt)

#%%

for i in np.arange(0,16):
    lsm_prest = LSM(z, x, t, sources[:,i:i+1], recs, v0, wav, wavc,
              mode='analytic')
    madj_prest = lsm_prest.Demop.H * d[i,:,:].ravel()
    madj_prest = madj_prest.reshape(nx, nz)

plt.figure(figsize=(10,5))
im = plt.imshow(madj_prest.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('madj_prest')

#%%
import logging
import numpy as np
import skfmm

from scipy.sparse.linalg import lsqr
from pylops import Spread
from pylops.signalprocessing import Convolve1D
from pylops.utils import dottest as Dottest

try:
    import skfmm
except ModuleNotFoundError:
    skfmm = None
    skfmm_message = 'Skfmm package not installed. Choose method=analytical ' \
                    'if using constant velocity or run ' \
                    '"pip install scikit-fmm" or ' \
                    '"conda install -c conda-forge scikit-fmm".'
except Exception as e:
    skfmm = None
    skfmm_message = 'Failed to import skfmm (error:%s).' % e

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

# def Demigration(z, x, t, srcs, recs, vel, wav, wavcenter,y=None, trav=None, mode='eikonal'):
# set the var names:
srcs = sources; vel = v0; wavcenter = wavc; y = None; mode='analytic';                     


ndim, _, dims, ny, nx, nz, ns, nr, _, _, _, _, _ = \
        _identify_geometry(z, x, srcs, recs, y=None)
dt = t[1] - t[0]
nt = len(t)

# if mode in ['analytic', 'eikonal']:
# compute traveltime table
trav_tmp = _traveltime_table(z, x, srcs, recs, vel, y=y, mode=mode)[0]

itrav_tmp = (trav_tmp / dt).astype('int32')
travd_tmp = (trav_tmp / dt - itrav_tmp)
if ndim == 2:
    itrav_tmp = itrav_tmp.reshape(nx, nz, ns * nr)
    travd_tmp = travd_tmp.reshape(nx, nz, ns * nr)
    dims = tuple(dims)
else:
    itrav_tmp = itrav_tmp.reshape(ny*nx, nz, ns * nr)
    travd_tmp = travd_tmp.reshape(ny*nx, nz, ns * nr)
    dims = (dims[0]*dims[1], dims[2])

# create operator
sop = Spread(dims=dims, dimsd=(ns * nr, nt), table=itrav_tmp, dtable=travd_tmp, 
             engine='numba')

cop = Convolve1D(ns * nr * nt, h=wav, offset=wavcenter, dims=(ns * nr, nt),
                 dir=1)
demop = cop * sop


#%%
madj = lsm.Demop.H * d.ravel()
madj = madj.reshape(nx, nz)

plt.figure(figsize=(10,5))
im = plt.imshow(madj.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('madj')

# demigration
dadj = lsm.Demop * madj.ravel()
dadj = dadj.reshape(ns, nr, nt)

#%%
fig, axs = plt.subplots(1, 2, figsize=(5, 4))
axs[0].imshow(d[ns//2, :, :300].T, cmap='gray')
axs[0].set_title(r'$d$')
axs[0].axis('tight')
axs[1].imshow(dadj[ns//2, :, :300].T, cmap='gray')
axs[1].set_title(r'$d_{adj}$')
axs[1].axis('tight')