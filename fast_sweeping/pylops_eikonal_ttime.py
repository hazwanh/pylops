#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:56:12 2021

@author: csi-13
"""

# trav, trav_srcs, trav_recs = _traveltime_table(z, x, sources, recs, vel, mode='eikonal')   

srcs = sources
y = None

ndims, shiftdim, _, ny, nx, nz, ns, nr, _, _, _, dsamp, origin = \
    _identify_geometry(z, x, srcs, recs, y=y)   # calculate the geometry 
                
trav_srcs = np.zeros((ny * nx * nz, ns))        # create the empty matrix (1x375x375,31)                   
trav_recs = np.zeros((ny * nx * nz, nr))        # create the empty matrix (1x375x375,31)
for isrc, src in enumerate(srcs.T):
    src = np.round((src-origin)/dsamp).astype(np.int32) # 
    phi = np.ones_like(vel)
    if ndims == 2:
        phi[src[0], src[1]] = -1
    else:
        phi[src[0], src[1], src[2]] = -1
    trav_srcs[:, isrc] = (skfmm.travel_time(phi=phi,
                                            speed=vel,
                                            dx=dsamp)).ravel()
for irec, rec in enumerate(recs.T):
    rec = np.round((rec-origin)/dsamp).astype(np.int32)
    phi = np.ones_like(vel)
    if ndims == 2:
        phi[rec[0], rec[1]] = -1
    else:
        phi[rec[0], rec[1], rec[2]] = -1
    trav_recs[:, irec] = (skfmm.travel_time(phi=phi,
                                            speed=vel,
                                            dx=dsamp)).ravel()
trav = trav_srcs.reshape(ny * nx * nz, ns, 1) + \
       trav_recs.reshape(ny * nx * nz, 1, nr)
trav = trav.reshape(ny * nx * nz, ns * nr)