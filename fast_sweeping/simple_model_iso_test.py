#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:51:41 2021

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

#%% Create the velocity and reflectivity

# Velocity Model
nx, nz = 201, 100
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
nr = 25
rx = np.linspace(dx, nx-dx, nr)
# rx = np.linspace(dx, (nx)*dx, nr)
rz = 5*np.ones(nr)
recs = np.vstack((rx, rz))
dr = recs[0,1]-recs[0,0]

# Sources
ns = 25
sx = np.linspace(dx, nx-dx, nr)
# sx = np.linspace(dx, (nx)*dx, ns)
sz = 5*np.ones(ns)
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

#%% Run fast-sweeping

# Point-source location
zmin = min(z); xmin = min(x); int_dz = dz;
zmax = max(z); xmax = max(x); int_dx = dx;

Z,X = np.meshgrid(z,x,indexing='ij')

# add eta and epsilon to data
vz = vel.T #
epsilon_true = np.zeros((nx,nz))
delta_true = np.zeros((nx,nz))
theta_true = np.zeros((nx,nz))
vx = vz*np.sqrt(1+2*epsilon_true.T)
eta = (epsilon_true.T-delta_true.T)/(1+2*delta_true.T)
# eta = eta_true.T
theta =  theta_true.T

# Number of fast sweeping iterations
niter = 2

# Number of fixed point iterations 
nfpi = 5


TcompTotal = np.ones((nx*nz,len(sx)))
TcompTotal2 = np.ones((nx*nz,len(sx)))
TfacTot = np.zeros((nx*nz,len(sx)))
inx = nx
inz = nz


# Source indices
for i in range(0,len(sx)):
    isz = int(round((sz[i]-zmin)/dz))
    isx = int(round((sx[i]-xmin)/dx))

    print(f"Source number: {i+1} out of {len(sx)} ")
    print(f"Source location: (sz,sx) = ({sz[i]},{sx[i]})")
    print(f"Source indices: (isz,isx) = ({isz},{isx})")
    
    # Checking if the source-point does not lie on the computation grid
    # if abs(int(sz[i]/dz)-sz[i]/dz)>1e-8 or abs(int(sx[i]/dx)-sx[i]/dx)>1e-8:
    #     raise ValueError('Source point not on the computation grid \n Either reduce grid spacing or change the source location.')
    
    # Analytical solution for the known traveltime part
    
    # Velocity at the source location
    vzs = vz[int(isz),int(isx)]
    vxs = vx[int(isz),int(isx)]
    thetas = theta[int(isz),int(isx)]
    
    a0 = vxs**2*np.cos(thetas)**2 + vzs**2*np.sin(thetas)**2
    b0 = vzs**2*np.cos(thetas)**2 + vxs**2*np.sin(thetas)**2
    c0 = np.sin(thetas)*np.cos(thetas)*(vzs**2-vxs**2)
    
    T0 = np.sqrt((b0*(X-sx[i])**2 + 2*c0*(X-sx[i])*(Z-sz[i]) 
                  + a0*(Z-sz[i])**2)/(a0*b0-c0**2)); 
    
    px0 = np.divide(b0*(X-sx[i]) + c0*(Z-sz[i]), 
                    T0*(a0*b0-c0**2), out=np.zeros_like(T0), where=T0!=0)
    pz0 = np.divide(a0*(Z-sz[i]) + c0*(X-sx[i]), 
                    T0*(a0*b0-c0**2), out=np.zeros_like(T0), where=T0!=0)
    
    # Initialize right hand side function to 1
    rhs = np.ones((nz,nx))
    
    # Placeholders to compute change in traveltime on each fixed-point iteration
    tn = np.zeros((nz,nx))
    tn1 = np.zeros((nz,nx))
    
    print(f'\nRunning for first-order regular fast sweeping method')
    # Initialize right hand side function to 1
    rhs = np.ones((nz,nx))
    
    # Placeholders to compute change in traveltime on each fixed-point iteration
    tn = np.zeros((nz,nx))
    tn1 = np.zeros((nz,nx))
    
    time_start = tm.time()
    for loop in range(nfpi):
        # Run the initializer
        T = ttieik.fastsweep_init2d(nz, nx, dz, dx, isz, isx, zmin, zmax)
    
        # Run the fast sweeping iterator
        ttieik.fastsweep_run2d(T, vz, vx, theta, niter, nz, nx, dz, dx, isz, isx, rhs)
        
        pz = np.gradient(T,dz,axis=0,edge_order=2)
        px = np.gradient(T,dx,axis=1,edge_order=2)
        
        pxdash = np.cos(theta)*px + np.sin(theta)*pz
        pzdash = np.cos(theta)*pz - np.sin(theta)*px
        
        rhs = 1 + ((2*eta*vx**2*vz**2)/(1+2*eta))*(pxdash**2)*(pzdash**2)
        
        tn1 = tn
        tn  = T
        print(f'L1 norm of update {np.sum(np.abs(tn1-tn))/(nz*nx)}')
    
    time_end = tm.time()
    print('FD modeling runtime:', (time_end - time_start), 's')
    
    Tcomp = T
    TcompTotal[:,i] = T.reshape(inz*inx)
    
    print(f'---------------------------------------- \n')

# save the travel time    
# io.savemat('TcompTotal_scattered_model_25x25.mat',{'TcompTotal':TcompTotal})
# TcompTotal = io.loadmat('TcompTotal_scattered_model_25x25.mat')['TcompTotal']    

tcomp_t = np.zeros(((int(nx))*(int(nz)),len(sx)))
for i in range(len(sx)):
    tcomp_new = (TcompTotal[:,i].reshape((int(nz)),(int(nx)))).T
    tcomp_t[:,i] = tcomp_new.reshape((int(nz))*(int(nx)))
    
ny = 1; ns=nr=len(sx)
trav_tcomp = tcomp_t.reshape((int(nz)) * (int(nx)), ns, 1) + \
       tcomp_t.reshape((int(nz)) * (int(nx)), 1, nr)
trav_tcomp = trav_tcomp.reshape(ny * (int(nz)) * (int(nx)), ns * nr)

#%% Calculate the traveltime table using fast sweeping

itrav_fs = (np.floor(trav_tcomp/dt)).astype(np.int32)
travd_fs = (trav_tcomp/dt - itrav_fs)
itrav_fs = itrav_fs.reshape(nx, nz, ns*nr)
travd_fs = travd_fs.reshape(nx, nz, ns*nr)

#%% Generate lsm operator, data and madj for fast-sweeping
Sop_fs = Spread(dims=(nx, nz), dimsd=(ns*nr, nt), table=itrav_fs, dtable=travd_fs, engine='numba')
dottest(Sop_fs, ns*nr*nt, nx*nz)
Cop_fs = Convolve1D(ns*nr*nt, h=wav, offset=wavc, dims=(ns*nr, nt), dir=1)

LSMop_fs = Cop_fs*Sop_fs
LSMop_fs = LinearOperator(LSMop_fs, explicit=False)

d_fs = LSMop_fs * refl.ravel()
d_fs = d_fs.reshape(ns, nr, nt)

madj_fs = LSMop_fs.H * d_fs.ravel()
madj_fs = madj_fs.reshape(nx, nz)

#%% Calculate the travel time
trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vel, mode='eikonal')

#%% Create the wavelet
nt = 651
dt = 0.004
t = np.arange(nt)*dt
wav, wavt, wavc = ricker(t[:41], f0=20)

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

#%% Display the madj

rmin = -np.abs(madj_py).max()
rmax = np.abs(madj_py).max()

plt.figure(figsize=(10,5))
im = plt.imshow(madj_py.T, cmap='gray',vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('madj_py')

plt.figure(figsize=(10,5))
im = plt.imshow(madj_fs.T, cmap='gray',vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('madj_fs')

#%%
minv_py = LSMop_py.div(d_py.ravel(), niter=25)
minv_py = minv_py.reshape(nx, nz)

minv_fs = LSMop_fs.div(d_fs.ravel(), niter=25)
minv_fs = minv_fs.reshape(nx, nz)

#%% Display the migrated image

rmin = -np.abs(refl).max()
rmax = np.abs(refl).max()

plt.figure(figsize=(10,5))
im = plt.imshow(minv_fs.T, cmap='gray',vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('minv_fs')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_py.T, cmap='gray',vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('minv_py')
#%% Generate the contour plot

zmin = min(z); xmin = min(x);
zmax = max(z); xmax = max(x); 

# Traveltime contour plots
n = 312 # for 31:481, 60:1828 ((ns+1)*(ns/2))
trav_1 = trav[:,n].reshape(int(nx),int(nz))
trav_tcomp_1 = trav_tcomp[:,n].reshape(int(nx),int(nz))

plt.figure(figsize=(10,5))

ax = plt.gca()
im1 = ax.imshow(vx,extent = (x[0], x[-1], z[-1], z[0]), aspect=1, cmap="jet")
im2 = ax.contour(trav_1.T, 10, extent = [xmin,xmax,zmin,zmax], colors='g',linestyles = 'dashed')
im3 = ax.contour(trav_tcomp_1.T, 10, extent=[xmin,xmax,zmin,zmax], colors='r',linestyles = 'dashed')

ax.plot(sx,sz,'r*',markersize=8)

plt.xlabel('Offset (km)', fontsize=14)
plt.ylabel('Depth (km)', fontsize=14)
plt.title('Traveltime contour plot',fontsize=14)
plt.colorbar(im1)
ax.tick_params(axis='both', which='major', labelsize=8)
# plt.gca().invert_yaxis()
h1,_ = im2.legend_elements()
h2,_ = im3.legend_elements()
ax.legend([h1[0], h2[0]], ['pylops tt', 'fast-sweep tt'],fontsize=12,loc='lower right')

# ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
