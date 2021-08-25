#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:24:03 2021

@author: csi-13
"""

#%% Load module
%load_ext autoreload
%autoreload 2
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pywt

from scipy import misc
from pylops.utils                      import dottest
from pylops.utils.wavelets             import *
from pylops.utils.seismicevents        import *
from pylops.basicoperators             import *
from pylops.signalprocessing           import *
from pylops.waveeqprocessing.mdd       import *
from pylops.optimization.leastsquares  import *
from pylops.optimization.sparsity  import IRLS as IRLSpylops
from pylops.optimization.sparsity  import FISTA

#%% load the image and create the blurring operator
im = sp.misc.face()[::2, ::2, 0]

Nz, Nx = im.shape

# Blurring guassian operator
nh = [51, 51]
#hz = np.exp(-0.1*np.linspace(-(nh[0]//2), nh[0]//2, nh[0])**2)
#hx = np.exp(-0.03*np.linspace(-(nh[1]//2), nh[1]//2, nh[1])**2)
hz = sp.signal.gaussian(nh[0], 5, sym=True)
hx = sp.signal.gaussian(nh[1], 10, sym=True)
hz /= np.trapz(hz) # normalize the integral to 1
hx /= np.trapz(hx) # normalize the integral to 1
h = hz[:, np.newaxis] * hx[np.newaxis, :]

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
him = ax.imshow(h)
ax.set_title('Blurring operator')
fig.colorbar(him, ax=ax)
ax.axis('tight')

#%% Create the blurred image and apply the deblurred operator

Cop = Convolve2D(Nz * Nx, h=h,
                 offset=(nh[0] // 2,
                         nh[1] // 2),
                 dims=(Nz, Nx), dtype='float32')
Wop = DWT2D((Nz, Nx), wavelet='haar', level=5)

imblur = Cop * im.flatten()

imdeblur = NormalEquationsInversion(Cop, None, imblur, maxiter=50)

imdeblurl1 = FISTA(Cop * Wop.H, imblur, eps=1e-1, niter=200, show=True)[0]
imdeblurl1 = Wop.H * imdeblurl1

imblur = imblur.reshape(Nz, Nx)
imdeblur = imdeblur.reshape(Nz, Nx)
imdeblurl1 = imdeblurl1.reshape(Nz, Nx)

#%% Display

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
ax1.imshow(im, cmap='gray', vmin=0, vmax=250)
ax1.axis('tight')
ax1.set_title('Original')
ax2.imshow(imblur, cmap='gray', vmin=0, vmax=250)
ax2.axis('tight')
ax2.set_title('Blurred');
ax3.imshow(imdeblur, cmap='gray', vmin=0, vmax=250)
ax3.axis('tight')
ax3.set_title('Deblurred L2');
ax4.imshow(imdeblurl1, cmap='gray', vmin=0, vmax=250)
ax4.axis('tight')
ax4.set_title('Deblurred L1');

#%%

