#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:39:01 2021

@author: csi-13
"""

#%%
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fseikonal.TTI.ttieikonal as ttieik
import fseikonal.TTI.facttieikonal as fttieik
import time as tm

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

#%%
for hby in [1]:


    print(f'Running with h/{hby}: \n')
    # Point-source location
    sx = np.linspace(0.05, 0.95, num=1)
    sz = np.ones(1)*0.05;

    zmin = 0.; zmax = 1.; dz = 0.01/hby;
    xmin = 0.; xmax = 1.; dx = 0.01/hby;

    z = np.arange(zmin,zmax+dz,dz)
    nz = z.size
    
    x = np.arange(xmin,xmax+dx,dx)
    nx = x.size
    
    # print(f'z and x shape is z = {z.shape}, x = {x.shape}')
    # print(f'z and x shape is dz = {dz}, dx = {dx}')

    Z,X = np.meshgrid(z,x,indexing='ij')

    v0 = 2.; # Velocity at the origin of the model
    vergrad = 1.5; # Vertical gradient
    horgrad = 0.5; # Horizontal gradient

    # Preparing velocity model
    vs = v0 + vergrad*sz + horgrad*sx # Velocity at the source location
    for i in range(0,len(sx)):
        vz = vs[i] + vergrad*(Z-sz[i]) + horgrad*(X-sx[i]);

    epsilon = np.zeros((nz,nx))
    delta = np.zeros((nz,nx))
    theta = np.zeros((nz,nx))
    
    # add eta and epsilon to data
    eta = np.zeros((nz,nx))
    vx = vz*np.sqrt(1+2*epsilon)

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
    
#%%
rx = sx; rz = sz;
sources = np.vstack((sx, sz)); recs = np.vstack((rx, rz));
trav, trav_srcs, trav_recs = _traveltime_table(z[::hby], x[::hby], sources, recs, vx[::hby,::hby], mode='eikonal')
#%%
trav_srcs_1 = trav[:,1].reshape(int(nx/hby)+1,int(nz/hby)+1)
#%%
# Plot the velocity model with the source location

plt.style.use('default')

plt.figure(figsize=(4,4))

ax = plt.gca()
im = ax.imshow(vz, extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")

ax.plot(sx,sz,'k*',markersize=8)

plt.xlabel('Offset (km)', fontsize=14)
plt.xticks(fontsize=10)

plt.ylabel('Depth (km)', fontsize=14)
plt.yticks(fontsize=10)

ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="6%", pad=0.15)

cbar = plt.colorbar(im, cax=cax)

cbar.set_label('km/s',size=10)
cbar.ax.tick_params(labelsize=10)

#%% Computing high-order solutions

# Second-order accuracy
#Trich2 = Tfac2 + (Tfac2-Tfac1)/3.0

# Third-order accuracy
# Tref = (16*Tfac4 - 8*Tfac2 + Tfac1)/9
TrefTot = ((16*TfacTot_4 - 8*TfacTot_2 + TfacTot_1)/9).reshape(101,101)
TrefTot = ((16*TfacTot_4[:,2] - 8*TfacTot_2[:,2] + TfacTot_1[:,2])/9).reshape(101,101)
TcompTotal_1 = TcompTotal[:,2].reshape(101,101)
#%% Plot the traveltime solution error

plt.style.use('default')

plt.figure(figsize=(4,4))

ax = plt.gca()
# im = ax.imshow(np.abs(Tref-Tcomp), extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")
# im = ax.imshow(np.abs(TrefTot-TcompTotal_1), extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")
# im = ax.imshow(np.abs(TcompTotal_1-TrefTot), extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")
im = ax.imshow(np.abs(TcompTotal_1-TrefTot), extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")


plt.xlabel('Offset (km)', fontsize=14)
plt.xticks(fontsize=10)

plt.ylabel('Depth (km)', fontsize=14)
plt.yticks(fontsize=10)

ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

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
# im1 = ax.contour(TrefTot, 6, extent=[xmin,xmax,zmin,zmax], colors='k')
# im2 = ax.contour(TcompTotal_1, 6, extent=[xmin,xmax,zmin,zmax], colors='r',linestyles = 'dashed')

ax.plot(sx,sz,'k*',markersize=8)

plt.xlabel('Offset (km)', fontsize=14)
plt.ylabel('Depth (km)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.gca().invert_yaxis()
h1,_ = im1.legend_elements()
h2,_ = im2.legend_elements()
ax.legend([h1[0], h2[0]], ['Reference', 'First-order FSM'],fontsize=12)

ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

#%% Compared with pylops TT

plt.figure(figsize=(4,4))

ax = plt.gca()
# im1 = ax.contour(Tref, 6, extent=[xmin,xmax,zmin,zmax], colors='k')
# im2 = ax.contour(Tcomp, 6, extent=[xmin,xmax,zmin,zmax], colors='r',linestyles = 'dashed')
im1 = ax.contour(TrefTot, 6, extent=[xmin,xmax,zmin,zmax], colors='k')
im2 = ax.contour(TcompTotal_1, 6, extent=[xmin,xmax,zmin,zmax], colors='r',linestyles = 'dashed')
im3 = ax.contour(trav_srcs_1.T, 6, extent=[xmin,xmax,zmin,zmax], colors='b',linestyles = 'dotted')

ax.plot(sx,sz,'k*',markersize=8)

plt.xlabel('Offset (km)', fontsize=14)
plt.ylabel('Depth (km)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.gca().invert_yaxis()
h1,_ = im1.legend_elements()
h2,_ = im2.legend_elements()
h3,_ = im3.legend_elements()
ax.legend([h1[0], h2[0],h3[0]], ['Reference', 'First-order FSM','pylops'],fontsize=12)

ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

#%%

ny = 1; ns=nr=len(sx)
trav_tcomp = tcomp_t.reshape((int(nx/hby)+1) * (int(nz/hby)+1), ns, 1) + \
       tcomp_t.reshape((int(nx/hby)+1) * (int(nz/hby)+1), 1, nr)
trav_tcomp = trav_tcomp.reshape(ny * (int(nx/hby)+1) * (int(nz/hby)+1), ns * nr)

tcomp_t = np.zeros(((int(nx/hby)+1)*(int(nz/hby)+1),len(sx)))
for i in range(len(sx)):
    tcomp_new = (TcompTotal[:,i].reshape((int(nx/hby)+1),(int(nz/hby)+1))).T
    tcomp_t[:,i] = tcomp_new.reshape((int(nx/hby)+1)*(int(nz/hby)+1))
    
ny = 1; ns=nr=len(sx)
trav_tcomp = tcomp_t.reshape((int(nx/hby)+1) * (int(nz/hby)+1), ns, 1) + \
       tcomp_t.reshape((int(nx/hby)+1) * (int(nz/hby)+1), 1, nr)
trav_tcomp = trav_tcomp.reshape(ny * (int(nx/hby)+1) * (int(nz/hby)+1), ns * nr)


#%%
n = 0
trav_1 = trav[:,n].reshape(int(nx/hby)+1,int(nz/hby)+1)
trav_tcomp_1 = trav_tcomp[:,n].reshape(int(nx/hby)+1,int(nz/hby)+1)
trav_srcs_1 = trav_srcs[:,n].reshape(int(nx/hby)+1,int(nz/hby)+1)
trav_recs_1 = trav_recs[:,n].reshape(int(nx/hby)+1,int(nz/hby)+1)
tcomp_t_1 = tcomp_t[:,n].reshape(int(nx/hby)+1,int(nz/hby)+1)
TrefTot = ((16*TfacTot_4[:,n] - 8*TfacTot_2[:,n] + TfacTot_1[:,n])/9).reshape(101,101)
TcompTotal_1 = TcompTotal[:,n].reshape(101,101)

plt.figure(figsize=(4,4))

ax = plt.gca()
# im1 = ax.contour(Tref, 6, extent=[xmin,xmax,zmin,zmax], colors='k')
# im2 = ax.contour(Tcomp, 6, extent=[xmin,xmax,zmin,zmax], colors='r',linestyles = 'dashed')
im1 = ax.contour(TrefTot.T, 6, extent=[xmin,xmax,zmin,zmax], colors='k')
im2 = ax.contour(TcompTotal_1, 6, extent=[xmin,xmax,zmin,zmax], colors='r',linestyles = 'dashed')
im4 = ax.contour(trav_1, 6, extent=[xmin,xmax,zmin,zmax], colors='b',linestyles = 'dotted')
im3 = ax.contour(trav_srcs_1, 6, extent=[xmin,xmax,zmin,zmax], colors='g',linestyles = 'dashed')
# im4 = ax.contour(tcomp_t_1.T, 6, extent=[xmin,xmax,zmin,zmax], colors='b',linestyles = 'dashed')
im5 = ax.contour(trav_tcomp_1.T, 6, extent=[xmin,xmax,zmin,zmax], colors='c',linestyles = 'dotted')

ax.plot(sx,sz,'k*',markersize=8)

plt.xlabel('Offset (km)', fontsize=14)
plt.ylabel('Depth (km)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.gca().invert_yaxis()
h1,_ = im1.legend_elements()
h2,_ = im2.legend_elements()
h3,_ = im3.legend_elements()
h4,_ = im4.legend_elements()
h5,_ = im5.legend_elements()
# ax.legend([h1[0], h2[0],h3[0],h4[0]], ['Reference', 'First-order FSM','pylops_trav','pylops_srcs'],fontsize=8)
# ax.legend([h1[0], h2[0],h3[0]], ['Reference', 'First-order FSM','pylops_srcs'],fontsize=12)
# ax.legend([h1[0], h2[0],h3[0],h4[0]], ['Reference', 'First-order FSM','trav_srcs','trav_srcs'],fontsize=12)
ax.legend([h1[0], h2[0],h3[0],h4[0],h5[0]], ['Reference', 'First-order FSM','pylops_trav','pylops_srcs','FSM_trav'],fontsize=8)

ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

#%%
n = 8
trav_1 = trav[:,n].reshape(int(nx/hby)+1,int(nz/hby)+1)
trav_tcomp_1 = trav_tcomp[:,n].reshape(int(nx/hby)+1,int(nz/hby)+1)

plt.figure(figsize=(4,4))

ax = plt.gca()
# im1 = ax.contour(Tref, 6, extent=[xmin,xmax,zmin,zmax], colors='k')
# im2 = ax.contour(Tcomp, 6, extent=[xmin,xmax,zmin,zmax], colors='r',linestyles = 'dashed')
im1 = ax.contour(trav_1, 6, extent=[xmin,xmax,zmin,zmax], colors='b',linestyles = 'dotted')
im2 = ax.contour(trav_tcomp_1, 6, extent=[xmin,xmax,zmin,zmax], colors='r',linestyles = 'dashed')

ax.plot(sx,sz,'k*',markersize=8)

plt.xlabel('Offset (km)', fontsize=14)
plt.ylabel('Depth (km)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.gca().invert_yaxis()
h1,_ = im1.legend_elements()
h2,_ = im2.legend_elements()
# h5,_ = im3.legend_elements()
# ax.legend([h1[0], h2[0],h3[0],h4[0]], ['Reference', 'First-order FSM','pylops_trav','pylops_srcs'],fontsize=8)
ax.legend([h1[0], h2[0]], ['pylops_trav','fs_trav'],fontsize=12)
# ax.legend([h1[0], h2[0],h3[0],h4[0],h5[0]], ['Reference', 'First-order FSM','pylops_trav','pylops_srcs','FSM_trav'],fontsize=8)

ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)