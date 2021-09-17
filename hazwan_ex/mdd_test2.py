#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:32:24 2021

@author: csi-13
"""

ncp = get_array_module(d)
nt1 = nt

nfmax_allowed = int(np.ceil((nt1+1)/2))
# nfmax_allowed = int(np.ceil(nt))
nfmax = nfmax_allowed

# d2 = np.moveaxis(d, -1, 0) 

madj_hl = lsm.Demop.H * d.flatten()                    # (47940,)
madj_hl = madj_hl.reshape(nx, nz)    # (799,60)
# madj_hl = np.moveaxis(madj_hl, 0, -1)                 # (60,799)

minv_hl = lsqr(lsm.Demop, d.flatten(), **dict(damp=1e-10, iter_lim=20, show=1))[0]
minv_hl = minv_hl.reshape(nx, nz)                    # (799, 60)
minv_hl = np.moveaxis(minv, 0, -1)                                 # (60, 799)

wav1 = wav.copy()
wav1 = wav1[ncp.newaxis]
minv_hl = get_fftconvolve(d)(minv_hl, wav1, mode='same')

plt.figure(figsize=(10,5))
im = plt.imshow(minv_hl.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('migrated data, minv_hl')

plt.figure(figsize=(10,5))
im = plt.imshow(madj_hl.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('migrated data, madj_hl')

plt.figure(figsize=(10,5))
im = plt.imshow(wav1.T, cmap='gray')
plt.axis('tight')

imdeblur = NormalEquationsInversion(lsm.Demop.H, None,minv_hl.ravel(),maxiter=50)
imdb = imdeblur.reshape(ns, nr, nt)

mimdb = lsm.Demop.H * imdb.flatten()
mimdb = mimdb.reshape(nx, nz)

plt.figure(figsize=(10,5))
im = plt.imshow(madj_hl.T, cmap='gray')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]')
plt.title('migrated data, mimdb')