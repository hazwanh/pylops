#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:27:37 2021

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

#%% Generate the marmousi model and display

datapath = '/home/hazwanh/Documents/pylops/fast_sweeping/bp_model/models.mat'
vel_true = (io.loadmat(datapath)['vp']).T
epsilon_true = (io.loadmat(datapath)['epsilon']).T
delta_true = (io.loadmat(datapath)['delta']).T
theta_true = (io.loadmat(datapath)['theta']).T
x = np.arange(0,vel_true.shape[0])
z = np.arange(0,vel_true.shape[1])

# x = np.arange(0,np.max(x)-np.min(x)+4,4)
# z = np.arange(0,np.max(z)-np.min(z)+4,4)
nx, nz = len(x), len(z)
dx, dz = 4, 4

refl = np.diff(vel_true, axis=1)
refl = np.hstack([refl, np.zeros((nx, 1))])

# Smooth velocity
v0 = 1492 # initial velocity
nsmooth=30
vel = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel_true, axis=0)
vel = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel, axis=1)
# epsilon = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, epsilon_true, axis=0)
# epsilon = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, epsilon, axis=1)
# delta = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, delta_true, axis=0)
# delta = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, delta, axis=1)
# theta = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, theta_true, axis=0)
# theta = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, theta, axis=1)

# Receivers
nr = 25
# rx = np.linspace(dx*25, (nx-25)*dx, nr)
rx = np.linspace(dx, nx-dx, nr)
rz = 5*np.ones(nr)
recs = np.vstack((rx, rz))
dr = recs[0,1]-recs[0,0]

# Sources
ns = 25
# sx = np.linspace(dx*25, (nx-25)*dx, ns)
sx = np.linspace(dx, nx-dx, ns)
sz = 5*np.ones(ns)
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
im = plt.imshow(epsilon_true.T, cmap='jet',
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Epsilon')
plt.ylim(z[-1], z[0])

plt.figure(figsize=(10,5))
im = plt.imshow(delta_true.T, cmap='jet',
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Delta')
plt.ylim(z[-1], z[0])

# plt.figure(figsize=(10,5))
# im = plt.imshow(vx, cmap='jet',
#                 extent = (x[0], x[-1], z[-1], z[0]))
# plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
# plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
# plt.colorbar(im)
# plt.axis('tight')
# plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
# plt.title('Velocity overlay with epsilon')
# plt.ylim(z[-1], z[0])


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

plt.figure(figsize=(10,5))
im = plt.imshow(theta, cmap='jet',
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('theta')
plt.ylim(z[-1], z[0])
#%%

for hby in [1]:


    print(f'Running with h/{hby}: \n')
    # Point-source location
    zmin = min(z); xmin = min(x); int_dz = dz; dz = 4/hby;
    zmax = max(z); xmax = max(x); int_dx = dx; dx = 4/hby;
    
    Z,X = np.meshgrid(z,x,indexing='ij')
    
    # add eta and epsilon to data
    vz = vel_true.T # 
    vx = vz*np.sqrt(1+2*epsilon_true.T)
    eta = (epsilon_true.T-delta_true.T)/(1+2*delta_true.T)
    theta =  theta_true.T

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

# save the travel time    
# io.savemat('TcompTotal_scattered_model_25x25.mat',{'TcompTotal':TcompTotal})
# TcompTotal = io.loadmat('TcompTotal_scattered_model_25x25.mat')['TcompTotal']    

tcomp_t = np.zeros(((int(nx/hby))*(int(nz/hby)),len(sx)))
for i in range(len(sx)):
    tcomp_new = (TcompTotal[:,i].reshape((int(nz/hby)),(int(nx/hby)))).T
    tcomp_t[:,i] = tcomp_new.reshape((int(nz/hby))*(int(nx/hby)))
    
ny = 1; ns=nr=len(sx)
trav_tcomp = tcomp_t.reshape((int(nz/hby)) * (int(nx/hby)), ns, 1) + \
       tcomp_t.reshape((int(nz/hby)) * (int(nx/hby)), 1, nr)
trav_tcomp = trav_tcomp.reshape(ny * (int(nz/hby)) * (int(nx/hby)), ns * nr)

#%% Generate wavelet and other parameter
nt = 650
dt = 0.004
t = np.arange(nt)*dt

wav, wavt, wavc = ricker(t[:41], f0=20)

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

# d_fs = io.loadmat('d_madj_minv_fs_fault.mat')['d_fs']
# madj_fs = io.loadmat('d_madj_minv_fs_fault.mat')['madj_fs']
# minv_fs_50 = io.loadmat('d_madj_minv_fs_fault.mat')['minv_fs_50']
# minv_fs_25 = io.loadmat('d_madj_minv_fs_fault.mat')['minv_fs_25']

#%% Computes the travel time using eikonal
trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vz.T, mode='eikonal') 

# Generate the ricker wavelet
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


#%%
minv_py = LSMop_py.div(d_py.ravel(), niter=25)
minv_py = minv_py.reshape(nx, nz)

minv_fs = LSMop_fs.div(d_fs.ravel(), niter=25)
minv_fs = minv_fs.reshape(nx, nz)