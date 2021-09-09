#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:09:23 2021

@author: hazwanh
"""

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

#%% Part 0: Create the wavelet, model and operator

# Set the input parameters
par = {'ox':0,'dx':2,    'nx':60,
       'oy':0,'dy':2,    'ny':100,
       'ot':0,'dt':0.004,'nt':400,
       'f0': 20, 'nfmax': 210}

v       = 1500
t0_m    = [0.1]
theta_m = [0]
phi_m   = [0]
amp_m   = [1.]

t0_G    = [0.05,0.2,0.3]
theta_G = [0,0,0]
phi_G   = [0,0,0]
amp_G   = [1.,0.6,2.]


# Create axis
t,t2,x,y = makeaxis(par)

# Create wavelet
wav = ricker(t[:41], f0=par['f0'])[0]

# Generate model
m, mwav =  linear2d(x,t,v,t0_m,theta_m,amp_m,wav) # True refl

# Generate operator
G,Gwav = linear3d(x,y,t,v,t0_G,theta_G,phi_G,amp_G,wav)

# Add negative part to data and model
m     = np.concatenate((np.zeros((par['nx'], par['nt']-1)), m), axis=-1)
mwav  = np.concatenate((np.zeros((par['nx'], par['nt']-1)), mwav), axis=-1)
Gwav2 = np.concatenate((np.zeros((par['ny'], par['nx'], par['nt']-1)), Gwav), axis=-1)

#%% Part 1: Multi-dimensional Convolution

"""
The so-called multi-dimensional convolution (MDC) is a chained operator. 
It is composed of a forward Fourier transform a multi-dimensional integration
and an inverse Fourier transform

This operation can be discretized and performed by means of a linear operator

D = F^{H}RF

where:
        F is the Fourier transform applied along the time axis 
        R is multi-dimensional convolution kernel.
"""

# Create data using MDC

Gwav_fft = np.fft.rfft(Gwav2, 2*par['nt']-1, axis=-1)
Gwav_fft = Gwav_fft[...,:par['nfmax']]

# create the operator
MDCop1=MDC(Gwav_fft.transpose(2,0,1), nt=2*par['nt']-1, nv=1, dt=par['dt'], dr=par['dx'], 
           twosided=True, transpose=False)
dottest(MDCop1, MDCop1.shape[0], MDCop1.shape[1], complexflag=3, verb=True);

# Create data
d = MDCop1*m.flatten()
d1 = MDCop1*m.T.flatten()

d = d.reshape(par['ny'], 2*par['nt']-1)
d1 = d1.reshape(2*par['nt']-1, par['ny']) # the synthetic data?

# Plotting

# Display the true model
plt.figure()
plt.imshow(mwav.T,aspect='auto',interpolation='nearest', vmin=-2, vmax=2, cmap='gray',
           extent=(x.min(),x.max(),t2.max(),t2.min()))
plt.title('true m', fontsize=15)
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()

# Display the generated d using mdc
plt.figure()
plt.imshow(d1,aspect='auto',interpolation='nearest', vmin=-20, vmax=20, cmap='gray',
           extent=(x.min(),x.max(),t2.max(),t2.min()))
plt.title('generated d through mdc', fontsize=15)
plt.xlabel('x'),plt.ylabel('t')
plt.tight_layout()

# # Display the G (inline and xline)
# plt.figure()
# plt.subplot(121)
# plt.imshow(Gwav2[int(par['ny']/2)].T,aspect='auto',interpolation='nearest', vmin=-2, vmax=2, cmap='gray',
#            extent=(x.min(),x.max(),t2.max(),t2.min()))
# plt.title('G - inline view', fontsize=15)
# plt.xlabel('x'),plt.ylabel('t')

# plt.subplot(122)
# plt.imshow(Gwav2[:,int(par['nx']/2)].T,aspect='auto',interpolation='nearest', vmin=-2, vmax=2, cmap='gray',
#            extent=(y.min(),y.max(),t2.max(),t2.min()))
# plt.title('G - xline view', fontsize=15)
# plt.xlabel('y'),plt.ylabel('t')
# plt.tight_layout()
#%% Part 2: Multi-dimensional Deconvolution

"""
MDD is an ill-solved problem. It aims is to remove the effect of the 
multidimensional convolution kernel or the so-called point-spread function 
(PSF).

It can be written as:
    d = Dm
    
or equivalently, by means of its normal equation:
    m = (D^{H}D)^{-1}D^{H}d

where D^{H}D is generally referred to as blurring operator or PSF.
"""

# invert the MDC operator
madj = MDCop1.H*d1.flatten()
minv, istop, itn, r1norm, r2norm = lsqr(MDCop1, d1.flatten(), damp=1e-10, iter_lim=10, show=1)[0:5]

madj = madj.reshape(2*par['nt']-1, par['nx']) # migrated equivalent?
minv = minv.reshape(2*par['nt']-1, par['nx']) # inverted m?

#%% Plotting

# Display
fig = plt.figure(figsize=(10,6))
ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
ax2 = plt.subplot2grid((1, 5), (0, 2), colspan=2)
ax3 = plt.subplot2grid((1, 5), (0, 4), colspan=2)

# Display the true model
ax1.imshow(mwav.T,aspect='auto',interpolation='nearest', cmap='gray',
           vmin=-mwav.max(), vmax=mwav.max(),
           extent=(x.min(),x.max(),t2.max(),t2.min()))
ax1.set_title('true m', fontsize=15)
ax1.set_xlabel('x'),ax1.set_ylabel('t')

# Display the adjoint (migrated) m
ax2.imshow(madj, aspect='auto',interpolation='nearest', cmap='gray', 
           vmin=-madj.max(), vmax=madj.max(),
           extent=(x.min(),x.max(),t2.max(),t2.min()))
ax2.set_title('adjoint m', fontsize=15)
ax2.set_xlabel('x'),ax1.set_ylabel('t')

# Display the inverted m
ax3.imshow(minv, aspect='auto', interpolation='nearest', cmap='gray',
           vmin=-minv.max(), vmax=minv.max(),
           extent=(x.min(),x.max(),t2.max(),t2.min()))
ax3.set_title('inverted m', fontsize=15)
ax3.set_xlabel('x'),ax1.set_ylabel('t')

fig.tight_layout()

# ax3.plot(madj[:, int(par['nx']/2)]/np.abs(madj[:, int(par['nx']/2)]).max(), t2, 'r', lw=5)
# ax3.plot(minv[:, int(par['nx']/2)]/np.abs(minv[:, int(par['nx']/2)]).max(), t2, '--k', lw=3)
# ax3.set_ylim([t2[-1],t2[0]])

#%% Preconditioned inversion

"""
We solve now the same problem with a preconditioning

d=DPm

where P is a masking operator that sets values in the negative part of the 
time axis equal to zero. 
This is added here as we know that our solution should be null in the negative 
time axis and it can be used to speed up convergence.
"""
P = np.ones((par['nx'],par['nt']*2-1))
P[:,:par['nt']-1]=0
Pop = Diagonal(P)

minv_prec= PreconditionedInversion(MDCop1, Pop, d1.flatten(), returninfo=True,
                                   **dict(damp=1e-10, iter_lim=10, show=1))[0]

minv_prec = minv_prec.reshape(2*par['nt']-1, par['nx'])

# Plotting

fig = plt.figure(figsize=(10,6))
ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
ax2 = plt.subplot2grid((1, 5), (0, 2), colspan=2)
ax3 = plt.subplot2grid((1, 5), (0, 4), colspan=2)

# Display the true model
ax1.imshow(mwav.T,aspect='auto',interpolation='nearest', cmap='gray',
           vmin=-mwav.max(), vmax=mwav.max(),
           extent=(x.min(),x.max(),t2.max(),t2.min()))
ax1.set_title('true m', fontsize=15)
ax1.set_xlabel('x'),ax1.set_ylabel('t')

# Display the adjoint (migrated) m
ax2.imshow(madj, aspect='auto',interpolation='nearest', cmap='gray', 
           vmin=-madj.max(), vmax=madj.max(),
           extent=(x.min(),x.max(),t2.max(),t2.min()))
ax2.set_title('adjoint m', fontsize=15)
ax2.set_xlabel('x'),ax1.set_ylabel('t')

# Display the inverted m
ax3.imshow(minv_prec,aspect='auto',interpolation='nearest', cmap='gray',
           vmin=-minv_prec.max(), vmax=minv_prec.max(),
           extent=(x.min(),x.max(),t2.max(),t2.min()))
ax3.set_title('precond inv m', fontsize=15)
ax3.set_xlabel('x'),ax1.set_ylabel('t')

fig.tight_layout()

# ax3.plot(madj[:, int(par['nx']/2)]/np.abs(madj[:, int(par['nx']/2)]).max(), t2, 'r', lw=5, label='Adj')
# ax3.plot(minv[:, int(par['nx']/2)]/np.abs(minv[:, int(par['nx']/2)]).max(), t2, '--g', lw=3, label='Inv')
# ax3.plot(minv_prec[:, int(par['nx']/2)]/np.abs(minv_prec[:, int(par['nx']/2)]).max(), t2, '--k', lw=3, label='Inv prec')
# ax3.set_ylim([t2[-1],t2[0]])
# ax3.legend()

#%% High level MDD routine

minv_hl,madj_hl,psfinv_hl,psfadj_hl = MDD(Gwav2, d1.T, 
                              dt=par['dt'], dr=par['dx'], nfmax=799, twosided=True, add_negative=False,
                              adjoint=True, psf=True, dtype='complex64', dottest=True, 
                              **dict(damp=1e-10, iter_lim=20, show=1))
# Display
fig = plt.figure(figsize=(10,6))
ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
ax2 = plt.subplot2grid((1, 5), (0, 2), colspan=2)
ax3 = plt.subplot2grid((1, 5), (0, 4), colspan=2)

# Display the true model
ax1.imshow(mwav.T,aspect='auto',interpolation='nearest', cmap='gray',
           vmin=-mwav.max(), vmax=mwav.max(),
           extent=(x.min(),x.max(),t2.max(),t2.min()))
ax1.set_title('true m', fontsize=15)
ax1.set_xlabel('x'),ax1.set_ylabel('t')

# Display the adjoint (migrated) m
ax2.imshow(madj_hl.T, aspect='auto',interpolation='nearest', cmap='gray', 
           vmin=-madj_hl.max(), vmax=madj_hl.max(),
           extent=(x.min(),x.max(),t2.max(),t2.min()))
ax2.set_title('adj m hl', fontsize=15)
ax2.set_xlabel('x'),ax1.set_ylabel('t')

# Display the inverted m
ax3.imshow(minv_hl.T, aspect='auto', interpolation='nearest', cmap='gray',
           vmin=-minv_hl.max(), vmax=minv_hl.max(),
           extent=(x.min(),x.max(),t2.max(),t2.min()))
ax3.set_title('inv m hl', fontsize=15)
ax3.set_xlabel('x'),ax1.set_ylabel('t')

fig.tight_layout()

#%% For preconditioner
minv_hl_prec,madj_hl_prec = MDD(Gwav2, d1.T, 
                dt=par['dt'], dr=par['dx'], nfmax=799, twosided=True, add_negative=False,
                causality_precond=True, adjoint=True, psf=False, 
                dtype='complex64', dottest=True, 
                **dict(damp=1e-10, iter_lim=10, show=1))

# Display the true model
ax1.imshow(mwav.T,aspect='auto',interpolation='nearest', cmap='gray',
           vmin=-mwav.max(), vmax=mwav.max(),
           extent=(x.min(),x.max(),t2.max(),t2.min()))
ax1.set_title('true m', fontsize=15)
ax1.set_xlabel('x'),ax1.set_ylabel('t')

# Display the adjoint (migrated) m
ax2.imshow(madj_hl_prec.T, aspect='auto',interpolation='nearest', cmap='gray', 
           vmin=-madj_hl_prec.max(), vmax=madj_hl_prec.max(),
           extent=(x.min(),x.max(),t2.max(),t2.min()))
ax2.set_title('adj m hl prec', fontsize=15)
ax2.set_xlabel('x'),ax1.set_ylabel('t')

# Display the inverted m
ax3.imshow(minv_hl_prec.T, aspect='auto', interpolation='nearest', cmap='gray',
           vmin=-minv_hl_prec.max(), vmax=minv_hl_prec.max(),
           extent=(x.min(),x.max(),t2.max(),t2.min()))
ax3.set_title('inv m hl prec', fontsize=15)
ax3.set_xlabel('x'),ax1.set_ylabel('t')

fig.tight_layout()
