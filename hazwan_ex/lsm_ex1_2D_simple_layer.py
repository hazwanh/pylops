# -*- coding: utf-8 -*-
"""
Spyder Editor

Exercise 1: Simple model with 2 interfaces

@author: csi-13
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr

import pylops

plt.close('all')
np.random.seed(0)

#%%

# Velocity Model
nx, nz = 81, 60
dx, dz = 4, 4
x, z = np.arange(nx)*dx, np.arange(nz)*dz
v0 = 1000 # initial velocity
kv = 0. # gradient
vel = np.outer(np.ones(nx), v0 +kv*z)

# Reflectivity Model
refl = np.zeros((nx, nz))
refl[:, 30] = -1
refl[:, 50] = 0.5

# Receivers
nr = 11
rx = np.linspace(10*dx, (nx-10)*dx, nr)
rz = 20*np.ones(nr)
recs = np.vstack((rx, rz))
dr = recs[0,1]-recs[0,0]

# Sources
ns = 10
sx = np.linspace(dx*10, (nx-10)*dx, ns)
sz = 10*np.ones(ns)
sources = np.vstack((sx, sz))
ds = sources[0,1]-sources[0,0]

#%%
plt.figure(figsize=(10, 5))
im = plt.imshow(vel.T, cmap='gray', extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('Velocity')
plt.xlim(x[0], x[-1])

plt.figure(figsize=(10, 5))
im = plt.imshow(refl.T, cmap='gray', extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('Reflectivity')
plt.xlim(x[0], x[-1])

#%%

# create ricker wavelet:
nt = 651
dt = 0.004
t = np.arange(nt)*dt
wav, wavt, wavc = pylops.utils.wavelets.ricker(t[:41], f0=20)

# display the wavelet
fig,axes = plt.subplots()
axes.plot(wav,wavt)
plt.show()
#%% solve seismic migration as inverse problem
lsm = pylops.waveeqprocessing.LSM(z, x, t, sources, recs, v0, wav, wavc,
                                  mode='analytic')

d = lsm.Demop * refl.ravel()
d = d.reshape(ns, nr, nt)

madj = lsm.Demop.H * d.ravel()
madj = madj.reshape(nx, nz)

# using LS solution
minv = lsm.solve(d.ravel(), solver=lsqr, **dict(iter_lim=100))
minv = minv.reshape(nx, nz)

# using LS solution with sparse model (FISTA)
minv_sparse = lsm.solve(d.ravel(),
                        solver=pylops.optimization.sparsity.FISTA,
                        **dict(eps=1e2, niter=100))
minv_sparse = minv_sparse.reshape(nx, nz)

#%% demigration

# get the adjoint
dadj = lsm.Demop * madj.ravel()
dadj = dadj.reshape(ns, nr, nt)

# using LS solution
dinv = lsm.Demop * minv.ravel()
dinv = dinv.reshape(ns, nr, nt)

# using LS solution with sparse model (FISTA)
dinv_sparse = lsm.Demop * minv_sparse.ravel()
dinv_sparse = dinv_sparse.reshape(ns, nr, nt)

#%% Display the results:
    
# Part 1: stack
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0][0].imshow(refl.T, cmap='gray', vmin=-1, vmax=1)
axs[0][0].axis('tight')
axs[0][0].set_title(r'$m$')
axs[0][1].imshow(madj.T, cmap='gray', vmin=-madj.max(), vmax=madj.max())
axs[0][1].set_title(r'$m_{adj}$')
axs[0][1].axis('tight')
axs[1][0].imshow(minv.T, cmap='gray', vmin=-1, vmax=1)
axs[1][0].axis('tight')
axs[1][0].set_title(r'$m_{inv}$')
axs[1][1].imshow(minv_sparse.T, cmap='gray', vmin=-1, vmax=1)
axs[1][1].axis('tight')
axs[1][1].set_title(r'$m_{FISTA}$')

# Part 2: near shot
fig, axs = plt.subplots(1, 4, figsize=(10, 4))
axs[0].imshow(d[0, :, :300].T, cmap='gray',
              vmin=-d.max(), vmax=d.max())
axs[0].set_title(r'$d$')
axs[0].axis('tight')
axs[1].imshow(dadj[0, :, :300].T, cmap='gray',
              vmin=-dadj.max(), vmax=dadj.max())
axs[1].set_title(r'$d_{adj}$')
axs[1].axis('tight')
axs[2].imshow(dinv[0, :, :300].T, cmap='gray',
              vmin=-d.max(), vmax=d.max())
axs[2].set_title(r'$d_{inv}$')
axs[2].axis('tight')
axs[3].imshow(dinv_sparse[0, :, :300].T, cmap='gray',
              vmin=-d.max(), vmax=d.max())
axs[3].set_title(r'$d_{fista}$')
axs[3].axis('tight')

# Part 3: mid shot
fig, axs = plt.subplots(1, 4, figsize=(10, 4))
axs[0].imshow(d[ns//2, :, :300].T, cmap='gray',
              vmin=-d.max(), vmax=d.max())
axs[0].set_title(r'$d$')
axs[0].axis('tight')
axs[1].imshow(dadj[ns//2, :, :300].T, cmap='gray',
              vmin=-dadj.max(), vmax=dadj.max())
axs[1].set_title(r'$d_{adj}$')
axs[1].axis('tight')
axs[2].imshow(dinv[ns//2, :, :300].T, cmap='gray',
              vmin=-d.max(), vmax=d.max())
axs[2].set_title(r'$d_{inv}$')
axs[2].axis('tight')
axs[3].imshow(dinv_sparse[ns//2, :, :300].T, cmap='gray',
              vmin=-d.max(), vmax=d.max())
axs[3].set_title(r'$d_{fista}$')
axs[3].axis('tight')
