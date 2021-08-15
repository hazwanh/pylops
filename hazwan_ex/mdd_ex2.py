#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 01:40:18 2021

@author: hazwanh
"""
# Application 1: Multi-dimensional deconvolution (MDD)
#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline

#import warnings
#warnings.filterwarnings('ignore')
import cProfile
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


from scipy.sparse import csr_matrix, vstack
from scipy.linalg import lstsq, solve
from scipy.sparse.linalg import LinearOperator, cg, lsqr
from scipy import misc

from pylops.utils                      import dottest
from pylops.utils.wavelets             import *
from pylops.utils.seismicevents        import *
from pylops.utils.tapers               import *
from pylops.basicoperators             import *
from pylops.signalprocessing           import *
from pylops.waveeqprocessing.mdd       import *
from pylops.optimization.leastsquares  import *

#%% Multiple virtual sources
###### Input parameters
par = {'ox':0,'dx':2,    'nx':60,
       'oy':0,'dy':2,    'ny':100,
       'ot':0,'dt':0.004,'nt':300,
       'f0': 20, 'nfmax': 201}

v       = 1500
t0_m    = [0.2]
theta_m = [0]
phi_m   = [0]
amp_m   = [1.]

t0_G    = [0.1,0.2,0.3]
theta_G = [0,0,0]
phi_G   = [0,0,0]
amp_G   = [1.,0.6,2.]

# Create axis
t,t2,x,y = makeaxis(par)

# Create wavelet
wav = ricker(t[:41], f0=par['f0'])[0]

# Generate model
m, mwav =  linear3d(x,x,t,v,t0_m,theta_m,phi_m,amp_m,wav)

# Restrict number of virtual sources
nedge = 10
nv = par['nx'] - 2*nedge
if nedge > 0:
    m = m[:, nedge:-nedge]
    mwav = mwav[:, nedge:-nedge]

# Generate operator
G, Gwav = linear3d(x,y,t,v,t0_G,theta_G,phi_G,amp_G,wav)

# Add negative part to data and model
m     = np.concatenate((np.zeros((par['nx'], nv, par['nt']-1)),    m), axis=-1)
mwav  = np.concatenate((np.zeros((par['nx'], nv, par['nt']-1)), mwav), axis=-1)
Gwav2 = np.concatenate((np.zeros((par['ny'], par['nx'], par['nt']-1)), Gwav), axis=-1)

#%%
# Create operator in frequency domain
Gwav_fft = np.fft.rfft(Gwav2, 2*par['nt']-1, axis=-1)
Gwav_fft = Gwav_fft[...,:par['nfmax']]

MDCop1=MDC(Gwav_fft.transpose(2,0,1), nt=2*par['nt']-1, 
           nv=nv, dt=par['dt'], dr=par['dx'], 
           twosided=True, transpose=False)

m1 = m.transpose(2,0,1)
d1 = MDCop1*m1.flatten()
d1 = d1.reshape(2*par['nt']-1, par['ny'], nv)

#%% Invert for model (MDD)
minv,madj,psfinv,psfadj = MDD(Gwav2, d1.transpose(1, 2, 0), 
                              dt=par['dt'], dr=par['dx'], nfmax=par['nfmax'], twosided=True, 
                              add_negative=False, adjoint=True, psf=True, dtype='complex64', dottest=True, 
                              **dict(damp=1e-10, iter_lim=5, show=1))

#%% Plotting
plt.figure()
plt.subplot(121)
plt.imshow(mwav[int(par['nx']/2)].T,aspect='auto',interpolation='nearest', 
           vmin=-2, vmax=2, cmap='gray',
           extent=(x.min(),x.max(),t2.max(),t2.min()))
plt.title('m - inline view', fontsize=15)
plt.xlabel('x'),plt.ylabel('t')
plt.subplot(122)
plt.imshow(mwav[:,int(par['nx']/2),:].T,aspect='auto',interpolation='nearest', 
           vmin=-2, vmax=2, cmap='gray',
           extent=(y.min(),y.max(),t2.max(),t2.min()))
plt.title('m - xline view', fontsize=15)
plt.xlabel('y'),plt.ylabel('t')
plt.tight_layout()


plt.figure()
plt.subplot(121)
plt.imshow(Gwav2[int(par['ny']/2)].T,aspect='auto',interpolation='nearest', 
           vmin=-2, vmax=2, cmap='gray',
           extent=(x.min(),x.max(),t2.max(),t2.min()))
plt.title('G - inline view', fontsize=15)
plt.xlabel('x'),plt.ylabel('t')
plt.subplot(122)
plt.imshow(Gwav2[:,int(par['nx']/2)].T,aspect='auto',interpolation='nearest', 
           vmin=-2, vmax=2, cmap='gray',
           extent=(y.min(),y.max(),t2.max(),t2.min()))
plt.title('G - xline view', fontsize=15)
plt.xlabel('y'),plt.ylabel('t')
plt.tight_layout()


plt.figure()
plt.subplot(121)
plt.imshow(d1[:, int(par['ny']/2)],aspect='auto',interpolation='nearest', 
           vmin=-20, vmax=20, cmap='gray',
           extent=(x.min(),x.max(),t2.max(),t2.min()))
plt.title('d - inline view', fontsize=15)
plt.xlabel('x'),plt.ylabel('t')
plt.subplot(122)
plt.imshow(d1[:, :,int(par['nx']/2)],aspect='auto',interpolation='nearest', 
           vmin=-20, vmax=20, cmap='gray',
           extent=(y.min(),y.max(),t2.max(),t2.min()))
plt.title('d - xline view', fontsize=15)
plt.xlabel('y'),plt.ylabel('t')
plt.tight_layout()


plt.figure()
plt.subplot(121)
plt.imshow(madj[int(par['nx']/2)].T,aspect='auto',interpolation='nearest', 
           vmin=-madj.max(), vmax=madj.max(), cmap='gray',
           extent=(x.min(),x.max(),t2.max(),t2.min()))
plt.title('adjoint m - inline view', fontsize=15)
plt.xlabel('x'),plt.ylabel('t')
plt.subplot(122)
plt.imshow(madj[:,int(par['nx']/2)].T,aspect='auto',interpolation='nearest', 
           vmin=-madj.max(), vmax=madj.max(), cmap='gray',
           extent=(y.min(),y.max(),t2.max(),t2.min()))
plt.title('adjoint m - xline view', fontsize=15)
plt.xlabel('y'),plt.ylabel('t')
plt.tight_layout()


plt.figure()
plt.subplot(121)
plt.imshow(minv[int(par['nx']/2)].T,aspect='auto',interpolation='nearest', 
           vmin=-minv.max(), vmax=minv.max(), cmap='gray',
           extent=(x.min(),x.max(),t2.max(),t2.min()))
plt.title('inverted m - inline view', fontsize=15)
plt.xlabel('x'),plt.ylabel('t')

plt.subplot(122)
plt.imshow(minv[:,int(par['nx']/2)].T,aspect='auto',interpolation='nearest',            
           vmin=-minv.max(), vmax=minv.max(), cmap='gray',
           extent=(y.min(),y.max(),t2.max(),t2.min()))
plt.title('inverted m - xline view', fontsize=15)
plt.xlabel('y'),plt.ylabel('t')
plt.tight_layout() 
