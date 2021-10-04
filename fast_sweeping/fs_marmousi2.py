#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 17:18:13 2021

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

#Velocity
inputfile='../../pylops_notebooks/data/avo/poststack_model.npz'

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
nr = 5
rx = np.linspace(dx*25, (nx-25)*dx, nr)
# rx = np.linspace(dx, (nx)*dx, nr)
rz = 20*np.ones(nr)
recs = np.vstack((rx, rz))
dr = recs[0,1]-recs[0,0]

# Sources
ns = 5
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

#%%
for hby in [1]:


    print(f'Running with h/{hby}: \n')
    # Point-source location
    zmin = min(z); xmin = min(x); int_dz = dz; dz = 4/hby;
    zmax = max(z); xmax = max(x); int_dx = dx; dx = 4/hby;
    
    Z,X = np.meshgrid(z,x,indexing='ij')
    
    epsilon = 0.2*np.ones((nz,nx))
    delta = 0.1*np.ones((nz,nx))
    theta = 30.*np.ones((nz,nx))*(mt.pi/180)
    
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
    
        time_start = tm.time()
        for loop in range(nfpi):
            # Run the initializer
            tau = fttieik.fastsweep_init2d(nz, nx, dz, dx, isz, isx, zmin, zmax)
    
            # Run the fast sweeping iterator
            fttieik.fastsweep_run2d(tau, T0, pz0, px0, vz, vx, theta, niter, nz, nx, dz, dx, isz, isx, rhs)
            
            pz = T0*np.gradient(tau,dz,axis=0,edge_order=2) + tau*pz0
            px = T0*np.gradient(tau,dx,axis=1,edge_order=2) + tau*px0
            
            pxdash = np.cos(theta)*px + np.sin(theta)*pz
            pzdash = np.cos(theta)*pz - np.sin(theta)*px
            
            rhs = 1 + ((2*eta*vx**2*vz**2)/(1+2*eta))*(pxdash**2)*(pzdash**2)
            
            tn1 = tn
            tn  = tau*T0
            print(f'L1 norm of update {np.sum(np.abs(tn1-tn))/(nz*nx)}')
            
    
        Tfac = (tau*T0)[::hby,::hby]
        TfacTot[:,i] = Tfac.reshape(inx*inz)
        exec(f'Tfac{hby} = Tfac') # This will assign traveltimes to variables called Tfac1, Tfac2, and Tfac4
        exec(f'TfacTot_{hby} = TfacTot')
        
        time_end = tm.time()
        print('FD modeling runtime:', (time_end - time_start), 's')


        # if hby==1:
    
        #     print(f'\nRunning for first-order regular fast sweeping method')
        #     # Initialize right hand side function to 1
        #     rhs = np.ones((nz,nx))
    
        #     # Placeholders to compute change in traveltime on each fixed-point iteration
        #     tn = np.zeros((nz,nx))
        #     tn1 = np.zeros((nz,nx))
    
        #     time_start = tm.time()
        #     for loop in range(nfpi):
        #         # Run the initializer
        #         T = ttieik.fastsweep_init2d(nz, nx, dz, dx, isz, isx, zmin, zmax)
    
        #         # Run the fast sweeping iterator
        #         ttieik.fastsweep_run2d(T, vz, vx, theta, niter, nz, nx, dz, dx, isz, isx, rhs)
                
        #         pz = np.gradient(T,dz,axis=0,edge_order=2)
        #         px = np.gradient(T,dx,axis=1,edge_order=2)
                
        #         pxdash = np.cos(theta)*px + np.sin(theta)*pz
        #         pzdash = np.cos(theta)*pz - np.sin(theta)*px
                
        #         rhs = 1 + ((2*eta*vx**2*vz**2)/(1+2*eta))*(pxdash**2)*(pzdash**2)
                
        #         tn1 = tn
        #         tn  = T
        #         print(f'L1 norm of update {np.sum(np.abs(tn1-tn))/(nz*nx)}')
    
        #     time_end = tm.time()
        #     print('FD modeling runtime:', (time_end - time_start), 's')

        #     Tcomp = T
        #     TcompTotal[:,i] = T.reshape(inx*inz) 

    print(f'---------------------------------------- \n')

#%%
# Plot the velocity model with the source location

plt.style.use('default')

plt.figure(figsize=(10,5))

ax = plt.gca()
# im = ax.imshow(vz.T, extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")
im = ax.imshow(vz,extent = (x[0], x[-1], z[-1], z[0]), aspect=1, cmap="jet")

# ax.plot(sx,sz,'k*',markersize=8)
ax.plot(sx[0],sz[0],'k*',markersize=8)

plt.xlabel('Offset (km)', fontsize=14)
plt.xticks(fontsize=10)

plt.ylabel('Depth (km)', fontsize=14)
plt.yticks(fontsize=10)

# ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="6%", pad=0.15)

cbar = plt.colorbar(im, cax=cax)

cbar.set_label('km/s',size=10)
cbar.ax.tick_params(labelsize=10)

#%%
# Computing high-order solutions

# Second-order accuracy
#Trich2 = Tfac2 + (Tfac2-Tfac1)/3.0

# Third-order accuracy
Tref = (16*Tfac4 - 8*Tfac2 + Tfac1)/9
# Tref = ((tau*T0)/1000)/9

#%%
# Plot the traveltime solution error

plt.style.use('default')

plt.figure(figsize=(4,4))

ax = plt.gca()
im = ax.imshow(np.abs(Tref-Tcomp), extent = (x[0], x[-1], z[-1], z[0]), aspect=1, cmap="jet")


plt.xlabel('Offset (km)', fontsize=14)
plt.xticks(fontsize=10)

plt.ylabel('Depth (km)', fontsize=14)
plt.yticks(fontsize=10)

# ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="6%", pad=0.15)

cbar = plt.colorbar(im, cax=cax)

cbar.set_label('seconds',size=10)
cbar.ax.tick_params(labelsize=10)

#%%
# Traveltime contour plots

plt.figure(figsize=(4,4))

ax = plt.gca()
im1 = ax.contour(Tref, 6, extent=[xmin,xmax,zmin,zmax], colors='k')
im2 = ax.contour(Tcomp, 6, extent=[xmin,xmax,zmin,zmax], colors='r',linestyles = 'dashed')

ax.plot(sx,sz,'k*',markersize=8)

plt.xlabel('Offset (km)', fontsize=14)
plt.ylabel('Depth (km)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.gca().invert_yaxis()
h1,_ = im1.legend_elements()
h2,_ = im2.legend_elements()
ax.legend([h1[0], h2[0]], ['Reference', 'First-order FSM'],fontsize=12)

# ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

#%%
TfactTot_both = TfacTot_1.reshape(nx * nz, ns, 1)+  \
    TfacTot_1.reshape(nx * nz, 1, nr)
TfactTot_both = TfactTot_both.reshape(nx * nz, ns * nr)
trav = TfactTot_both

