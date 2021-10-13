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

datapath = '/home/csi-13/Documents/pylops/fast_sweeping/bp_model/BP_model_crop_int4.mat'
vel_true = io.loadmat(datapath)['model_vp1_int4']
epsilon = io.loadmat(datapath)['model_eps1_int4']
delta = io.loadmat(datapath)['model_del1_int4']
theta = io.loadmat(datapath)['model_thet1_int4']
x = io.loadmat(datapath)['x1']
z = io.loadmat(datapath)['z1']

x = np.arange(0,np.max(x)-np.min(x), 4)
z = np.arange(0,np.max(z)-np.min(z), 4)
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
nr = 3
rx = np.linspace(dx*25, (nx-25)*dx, nr)
# rx = np.linspace(dx, (nx)*dx, nr)
rz = 20*np.ones(nr)
recs = np.vstack((rx, rz))
dr = recs[0,1]-recs[0,0]

# Sources
ns = 3
sx = np.linspace(dx*25, (nx-25)*dx, ns)
# sx = np.linspace(dx, (nx)*dx, ns)
sz = 20*np.ones(ns)
sources = np.vstack((sx, sz))
ds = sources[0,1]-sources[0,0]

#%% Display the figure

velmin = 1492
velmax = np.abs(-1*vel_true).max()

plt.figure(figsize=(10,5))
im = plt.imshow(vel_true, cmap='jet', vmin = velmin, vmax = velmax,
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Velocity')
plt.ylim(z[-1], z[0])

plt.figure(figsize=(10,5))
im = plt.imshow(vel, cmap='jet', vmin = velmin, vmax = velmax, 
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
im = plt.imshow(refl, cmap='gray', vmin = reflmin, vmax = reflmax, 
                extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(recs[0],  recs[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(sources[0], sources[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('offset [m]'),plt.ylabel('depth [m]')
plt.title('Reflectivity')
plt.ylim(z[-1], z[0]);

#%%
for hby in [1]:


    print(f'Running with h/{hby}: \n')
    # Point-source location
    zmin = min(z); xmin = min(x); int_dz = dz; dz = 4/hby;
    zmax = max(z); xmax = max(x); int_dx = dx; dx = 4/hby;
    
    Z,X = np.meshgrid(z,x,indexing='ij')
    
    # add eta and epsilon to data
    vz = vel # 
    vx = vz*np.sqrt(1+2*epsilon)
    eta = (epsilon-delta)/(1+2*delta)

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
            TcompTotal[:,i] = T.reshape(inx*inz) 

    print(f'---------------------------------------- \n')

#%% Computes the travel time using eikonal
trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vel, mode='eikonal') 

#%% Perform LS on velocity model
nt = 600
dt = 0.004
t = np.arange(nt)*dt

# Generate the ricker wavelet
itrav = (np.floor(trav/dt)).astype(np.int32)
travd = (trav/dt - itrav)
itrav = itrav.reshape(nx, nz, ns*nr)
travd = travd.reshape(nx, nz, ns*nr)

wav, wavt, wavc = ricker(t[:41], f0=20)

#%%
Sop = Spread(dims=(nx, nz), dimsd=(ns*nr, nt), table=itrav, dtable=travd, engine='numba')
dottest(Sop, ns*nr*nt, nx*nz)
Cop = Convolve1D(ns*nr*nt, h=wav, offset=wavc, dims=(ns*nr, nt), dir=1)

LSMop = Cop*Sop
LSMop = LinearOperator(LSMop, explicit=False)

d = LSMop * refl.ravel()
d = d.reshape(ns, nr, nt)

madj = LSMop.H * d.ravel()
madj = madj.reshape(nx, nz)

# minv = LSMop.div(d.ravel(), niter=100)
# minv = minv.reshape(nx, nz)

#%%
rmin = -np.abs(refl).max()
rmax = np.abs(refl).max()

# true refl
plt.figure(figsize=(10,5))
im = plt.imshow(refl, cmap='gray', vmin=rmin, vmax=rmax)
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('true refl')

# madj
plt.figure(figsize=(10,5))
im = plt.imshow(madj, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('madj')