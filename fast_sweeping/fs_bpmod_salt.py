#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:21:31 2021

@author: csi-13
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

#%% Generate the marmousi model and display

datapath = '/home/csi-13/Documents/pylops/fast_sweeping/bp_model/bpmodel_salt_375x750_3750x6750.mat'
vel_true = (io.loadmat(datapath)['model_vp1_int4']).T
epsilon = (io.loadmat(datapath)['model_eps1_int4']).T
delta = (io.loadmat(datapath)['model_del1_int4']).T
theta = (io.loadmat(datapath)['model_thet1_int4']).T
x = io.loadmat(datapath)['x']
z = io.loadmat(datapath)['z']

x = np.arange(0,np.max(x)-np.min(x)+4,4)
z = np.arange(0,np.max(z)-np.min(z)+4,4)
nx, nz = len(x), len(z)
dx, dz = 4, 4

refl = np.diff(vel_true, axis=1)
refl = np.hstack([refl, np.zeros((nx, 1))])

# Smooth velocity
v0 = 1492 # initial velocity
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

velmin = 1492
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
im = plt.imshow(epsilon.T, cmap='jet',
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Epsilon')
plt.ylim(z[-1], z[0])

plt.figure(figsize=(10,5))
im = plt.imshow(delta.T, cmap='jet',
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Delta')
plt.ylim(z[-1], z[0])

plt.figure(figsize=(10,5))
im = plt.imshow(vx, cmap='jet',
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Velocity overlay with epsilon')
plt.ylim(z[-1], z[0])


plt.figure(figsize=(10,5))
im = plt.imshow(eta, cmap='jet',
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('eta')
plt.ylim(z[-1], z[0])
#%%

TcompTotal = io.loadmat('TcompTotal_salt_31x31_375x750.mat')['TcompTotal']
hby = 1;

for hby in [1]:


    print(f'Running with h/{hby}: \n')
    # Point-source location
    zmin = min(z); xmin = min(x); int_dz = dz; dz = 4/hby;
    zmax = max(z); xmax = max(x); int_dx = dx; dx = 4/hby;
    
    Z,X = np.meshgrid(z,x,indexing='ij')
    
    # add eta and epsilon to data
    vz = vel.T # 
    vx = vz*np.sqrt(1+2*epsilon.T)
    eta = (epsilon.T-delta.T)/(1+2*delta.T)
    theta =  theta.T

    # Number of fast sweeping iterations
    niter = 2

    # Number of fixed point iterations 
    nfpi = 5
    
    if hby == 1:
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
    
        # time_start = tm.time()
        # for loop in range(nfpi):
        #     # Run the initializer
        #     tau = fttieik.fastsweep_init2d(nz, nx, dz, dx, isz, isx, zmin, zmax)
    
        #     # Run the fast sweeping iterator
        #     fttieik.fastsweep_run2d(tau, T0, pz0, px0, vz, vx, theta, niter, nz, nx, dz, dx, isz, isx, rhs)
            
        #     pz = T0*np.gradient(tau,dz,axis=0,edge_order=2) + tau*pz0
        #     px = T0*np.gradient(tau,dx,axis=1,edge_order=2) + tau*px0
            
        #     pxdash = np.cos(theta)*px + np.sin(theta)*pz
        #     pzdash = np.cos(theta)*pz - np.sin(theta)*px
            
        #     rhs = 1 + ((2*eta*vx**2*vz**2)/(1+2*eta))*(pxdash**2)*(pzdash**2)
            
        #     tn1 = tn
        #     tn  = tau*T0
        #     print(f'L1 norm of update {np.sum(np.abs(tn1-tn))/(nz*nx)}')
            
    
        # Tfac = (tau*T0)[::hby,::hby]
        # TfacTot[:,i] = Tfac.reshape(inx*inz)
        # exec(f'Tfac{hby} = Tfac') # This will assign traveltimes to variables called Tfac1, Tfac2, and Tfac4
        # exec(f'TfacTot_{hby} = TfacTot')
        
        # time_end = tm.time()
        # print('FD modeling runtime:', (time_end - time_start), 's')


        if hby==1:
    
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

tcomp_t = np.zeros(((int(nx/hby))*(int(nz/hby)),len(sx)))
for i in range(len(sx)):
    tcomp_new = (TcompTotal[:,i].reshape((int(nz/hby)),(int(nx/hby)))).T
    tcomp_t[:,i] = tcomp_new.reshape((int(nz/hby))*(int(nx/hby)))
    
ny = 1; ns=nr=len(sx)
trav_tcomp = tcomp_t.reshape((int(nz/hby)) * (int(nx/hby)), ns, 1) + \
       tcomp_t.reshape((int(nz/hby)) * (int(nx/hby)), 1, nr)
trav_tcomp = trav_tcomp.reshape(ny * (int(nz/hby)) * (int(nx/hby)), ns * nr)

#%% 
nt = 800
dt = 0.004
t = np.arange(nt)*dt

# Generate the ricker wavelet
itrav_fs = (np.floor(trav_tcomp/dt)).astype(np.int32)
travd_fs = (trav_tcomp/dt - itrav_fs)
itrav_fs = itrav_fs.reshape(nx, nz, ns*nr)
travd_fs = travd_fs.reshape(nx, nz, ns*nr)

wav, wavt, wavc = ricker(t[:41], f0=20)

#%% 
Sop_fs = Spread(dims=(nx, nz), dimsd=(ns*nr, nt), table=itrav_fs, dtable=travd_fs, engine='numba')
dottest(Sop_fs, ns*nr*nt, nx*nz)
Cop_fs = Convolve1D(ns*nr*nt, h=wav, offset=wavc, dims=(ns*nr, nt), dir=1)

LSMop_fs = Cop_fs*Sop_fs
LSMop_fs = LinearOperator(LSMop_fs, explicit=False)

d_fs = LSMop_fs * refl.ravel()
d_fs = d_fs.reshape(ns, nr, nt)

madj_fs = LSMop_fs.H * d_fs.ravel()
madj_fs = madj_fs.reshape(nx, nz)


#%% Computes the travel time using eikonal
trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vx.T, mode='eikonal') 

# Generate the ricker wavelet
itrav_py = (np.floor(trav/dt)).astype(np.int32)
travd_py = (trav/dt - itrav_py)
itrav_py = itrav_py.reshape(nx, nz, ns*nr)
travd_py = travd_py.reshape(nx, nz, ns*nr)

#%%

Sop_py = Spread(dims=(nx, nz), dimsd=(ns*nr, nt), table=itrav_py, dtable=travd_py, engine='numba')
dottest(Sop_py, ns*nr*nt, nx*nz)
Cop_py = Convolve1D(ns*nr*nt, h=wav, offset=wavc, dims=(ns*nr, nt), dir=1)

LSMop_py = Cop_py*Sop_py
LSMop_py = LinearOperator(LSMop_py, explicit=False)

d_py = LSMop_py * refl.ravel()
d_py = d_py.reshape(ns, nr, nt)

madj_py = LSMop_py.H * d_py.ravel()
madj_py = madj_py.reshape(nx, nz)


#%%
minv_py = LSMop_py.div(d_py.ravel(), niter=25)
minv_py = minv_py.reshape(nx, nz)

minv_fs = LSMop_fs.div(d_fs.ravel(), niter=25)
minv_fs = minv_fs.reshape(nx, nz)

#%%
rmin = -np.abs(madj_fs).max()
rmax = np.abs(madj_fs).max()

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
rmin = -np.abs(refl).max()
rmax = np.abs(refl).max()

plt.figure(figsize=(10,5))
im = plt.imshow(refl.T, cmap='gray', vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('true refl')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_py.T, cmap='gray',vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('minv_py')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_fs.T, cmap='gray',vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('minv_fs')
#%%
zmin = min(z); xmin = min(x);
zmax = max(z); xmax = max(x); 

# Traveltime contour plots
n = 960
trav_1 = trav[:,n].reshape(int(nx/hby),int(nz/hby))
trav_tcomp_1 = trav_tcomp[:,n].reshape(int(nx/hby),int(nz/hby))

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
ax.legend([h1[0], h2[0]], ['pylops tt', 'fast-sweep tt'],fontsize=12)

# ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

#%% Generate shot gather
rmin = -np.abs(d_fs).max()
rmax = np.abs(d_fs).max()

fig, axs = plt.subplots(1, 3, figsize=(10, 6))
axs[0].imshow(d_py[0, :, :500].T, cmap='gray',vmin=rmin, vmax=rmax)
axs[0].set_title(f'shot: 1')
axs[0].axis('tight')
axs[1].imshow(d_py[ns//2, :, :500].T, cmap='gray',vmin=rmin, vmax=rmax)
axs[1].set_title(f'$shot:{ns//2} $')
axs[1].axis('tight')
axs[2].imshow(d_py[30, :, :500].T, cmap='gray',vmin=rmin, vmax=rmax)
axs[2].set_title(f'$shot: 31$')
axs[2].axis('tight')

fig, axs = plt.subplots(1, 3, figsize=(10, 6))
axs[0].imshow(d_fs[0, :, :500].T, cmap='gray',vmin=rmin, vmax=rmax)
axs[0].set_title(f'shot: 1')
axs[0].axis('tight')
axs[1].imshow(d_fs[ns//2, :, :500].T, cmap='gray',vmin=rmin, vmax=rmax)
axs[1].set_title(f'$shot:{ns//2} $')
axs[1].axis('tight')
axs[2].imshow(d_fs[30, :, :500].T, cmap='gray',vmin=rmin, vmax=rmax)
axs[2].set_title(f'$shot: 31$')
axs[2].axis('tight')