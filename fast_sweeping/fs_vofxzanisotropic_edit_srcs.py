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

#%%
for hby in [1,2,4]:


    print(f'Running with h/{hby}: \n')
    # Point-source location
    hby=1
    sx = np.linspace(0.05, 0.95, num=31)
    sz = np.ones(31)*0.05;

    zmin = 0.; zmax = 1.; dz = 0.01/hby;
    xmin = 0.; xmax = 1.; dx = 0.01/hby;

    z = np.arange(zmin,zmax+dz,dz)
    nz = z.size

    x = np.arange(xmin,xmax+dx,dx)
    nx = x.size

    Z,X = np.meshgrid(z,x,indexing='ij')

    v0 = 2.; # Velocity at the origin of the model
    vergrad = 1.5; # Vertical gradient
    horgrad = 0.5; # Horizontal gradient

    # Preparing velocity model
    vs = v0 + vergrad*sz + horgrad*sx # Velocity at the source location
    for i in range(0,len(sx)):
        vz = vs[i] + vergrad*(Z-sz[i]) + horgrad*(X-sx[i]);

    epsilon = 0.2*np.ones((nz,nx))
    delta = 0.1*np.ones((nz,nx))
    theta = 30.*np.ones((nz,nx))*(mt.pi/180)
    
    # add eta and epsilon to data
    eta = (epsilon-delta)/(1+2*delta)
    vx = vz*np.sqrt(1+2*epsilon)

    # Number of fast sweeping iterations
    niter = 2

    # Number of fixed point iterations 
    nfpi = 5

    # Source indices
    isz = np.zeros(len(sz))
    isx = np.zeros(len(sx))
    for i in range(0,len(sx)):
        isz[i] = int(round((sz[i]-zmin)/dz))
        isx[i] = int(round((sx[i]-xmin)/dx))

    print(f"Source location: (sz,sx) = ({sz},{sx})")
    print(f"Source indices: (isz,isx) = ({isz},{isx})")

    # Checking if the source-point does not lie on the computation grid
    for i in range(0,len(sx)):
        if abs(int(sz[i]/dz)-sz[i]/dz)>1e-8 or abs(int(sx[i]/dx)-sx[i]/dx)>1e-8:
            raise ValueError('Source point not on the computation grid \n Either reduce grid spacing or change the source location.')


    # Analytical solution for the known traveltime part

    # Velocity at the source location
    
    vzs = np.zeros(len(sx))
    vxs = np.zeros(len(sx))
    thetas  = np.zeros(len(sx))
    a0 = np.zeros(len(sx))
    b0 = np.zeros(len(sx))
    c0 = np.zeros(len(sx))
    for i in range(0,len(sx)):
        vzs[i] = vz[int(isz[i]),int(isx[i])]
        vxs[i] = vx[int(isz[i]),int(isx[i])]
        thetas[i] = theta[int(isz[i]),int(isx[i])]

        a0[i] = vxs[i]**2*np.cos(thetas[i])**2 + vzs[i]**2*np.sin(thetas[i])**2
        b0[i] = vzs[i]**2*np.cos(thetas[i])**2 + vxs[i]**2*np.sin(thetas[i])**2
        c0[i] = np.sin(thetas[i])*np.cos(thetas[i])*(vzs[i]**2-vxs[i]**2)

        T0 = np.sqrt((b0[i]*(X-sx[i])**2 + 2*c0[i]*(X-sx[i])*(Z-sz[i]) 
                      + a0[i]*(Z-sz[i])**2)/(a0[i]*b0[i]-c0[i]**2)); 

        px0 = np.divide(b0[i]*(X-sx[i]) + c0[i]*(Z-sz[i]), 
                        T0*(a0[i]*b0[i]-c0[i]**2), out=np.zeros_like(T0), where=T0!=0)
        pz0 = np.divide(a0[i]*(Z-sz[i]) + c0[i]*(X-sx[i]), 
                        T0*(a0[i]*b0[i]-c0[i]**2), out=np.zeros_like(T0), where=T0!=0)

        # Initialize right hand side function to 1
        rhs = np.ones((nz,nx))
    
        # Placeholders to compute change in traveltime on each fixed-point iteration
        tn = np.zeros((nz,nx))
        tn1 = np.zeros((nz,nx))
    
        time_start = tm.time()
        for loop in range(nfpi):
            # Run the initializer
            tau = fttieik.fastsweep_init2d(nz, nx, dz, dx, isz[i], isx[i], zmin, zmax)
    
            # Run the fast sweeping iterator
            fttieik.fastsweep_run2d(tau, T0, pz0, px0, vz, vx, theta, niter, nz, nx, dz, dx, isz[i], isx[i], rhs)
            
            pz = T0*np.gradient(tau,dz,axis=0,edge_order=2) + tau*pz0
            px = T0*np.gradient(tau,dx,axis=1,edge_order=2) + tau*px0
            
            pxdash = np.cos(theta)*px + np.sin(theta)*pz
            pzdash = np.cos(theta)*pz - np.sin(theta)*px
            
            rhs = 1 + ((2*eta*vx**2*vz**2)/(1+2*eta))*(pxdash**2)*(pzdash**2)
            
            tn1 = tn
            tn  = tau*T0
            print(f'L1 norm of update {np.sum(np.abs(tn1-tn))/(nz*nx)}')
            
    
        Tfac = (tau*T0)[::hby,::hby]
        exec(f'Tfac{hby} = Tfac') # This will assign traveltimes to variables called Tfac1, Tfac2, and Tfac4
    
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
                T = ttieik.fastsweep_init2d(nz, nx, dz, dx, isz[i], isx[i], zmin, zmax)
    
                # Run the fast sweeping iterator
                ttieik.fastsweep_run2d(T, vz, vx, theta, niter, nz, nx, dz, dx, isz[i], isx[i], rhs)
                
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
        Ttotal = Tcomp.reshape(nx*nz,i)

    print(f'---------------------------------------- \n')
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

#%%
# Computing high-order solutions

# Second-order accuracy
#Trich2 = Tfac2 + (Tfac2-Tfac1)/3.0

# Third-order accuracy
Tref = (16*Tfac4 - 8*Tfac2 + Tfac1)/9

#%%
# Plot the traveltime solution error

plt.style.use('default')

plt.figure(figsize=(4,4))

ax = plt.gca()
im = ax.imshow(np.abs(Tref-Tcomp), extent=[xmin,xmax,zmax,zmin], aspect=1, cmap="jet")


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

