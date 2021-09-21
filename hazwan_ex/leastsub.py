#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:40:19 2021

@author: csi-13
"""
import numpy as np
from scipy.signal import convolve2d

def correv(a, b, l=1):
    """

    %CORR  correlation of two arrays in first dimension
    %   C = CORR(A, B, L) correlates arrays A and B over length L
    %   in the first dimension
    %
    %   Author:  Eric Verschuur
    %   Date  :  Jan 1996

    """
    nta, nxa = a.shape
    ntb, nxb = b.shape
    
    # check size of b and a
    if nxa != nxb:
        print("a and b must have equal second dimension")
    
    # check output length to be small enough    
    if l > min(nta,ntb):
        print("output correlation must be smaller than length of a or b")
        
    # calculate correlation    
    nt = max(nta,ntb)
    aa = np.zeros((nt,nxa))
    bb = np.zeros((nt,nxb))
    c = np.zeros((l,nxa))
    aa[0:nta,:] = a
    bb[0:ntb,:] = b
    for j in range(0,l):
        c[j,:]  = sum(aa[j:nt,:]*bb[0:nt-j,:])

    return c

def conv2(x, y, mode='same'):
    """
    https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function
    """
    from scipy.signal import convolve2d
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def leastsub(inp,ref,lenf,eps):
    """
    %LEASTSUB - Least-squares subtraction
    %   [OUT,F]=leastsub(IN,REF,LENF,EPS); solves the system of equation:
    %      REF - F*IN = 0
    %   in the least-squares sense, such that E(REF - F*IN) = minimum.
    %   The matrix will be stabilized with EPS*EPS*max on main diagonal.
    %   The input file and reference should have equal first two dimensions.
    %   The filter length LENF determines the length of temporal filter F.
    %   The output array OUT contains the filtered version of the input file:
    %      OUT=F*IN
    %   If de input array has a third dimension > 1 it means that all input 
    %   matrices are subtracted simultaneously, according to:
    %      REF - F(1)*IN(1) - F(2)*IN(2) - ... = 0
    %   The output matrix OUT will contain the subsequent filtered input matrices:
    %     OUT(1)=F(1)*IN(1)
    %     OUT(2)=F(2)*IN(2)
    %     ....
    %
    %   Author: Eric Verschuur
    %   Date  : Aug 1997

    """
    from scipy.linalg import toeplitz
    
    if len(inp.shape) > 2:
        nt,nx,nin = inp.shape
    else:
        nt,nx = inp.shape; nin = 1

    ntr,nxr = ref.shape
    
    nrot = np.floor((lenf-1)/2)
    zz = np.zeros((int(nrot),nx))
    
    if len(inp.shape) > 2:
        xx = np.zeros(lenf,nin,nin)
        for j in range(0,nin):
            for i in range(0,nin):
                if i >= j:
                    xx[:,i,j] = np.transpose(sum(np.transpose(correv(inp[:,:,i],inp[:,:,j],lenf))))
                    xx[:,j,i] = np.transpose(sum(np.transpose(correv(inp[:,:,j],inp[:,:,i],lenf))))
                if i == 0:
                    Txxrow = toeplitz(xx[:,i,j],xx[:,j,i])
                else:
                    Txxrow = (np.hstack((Txxrow,toeplitz(np.hstack((xx[:,i,j],xx[:,j,i]))))))
                    
            yx = np.transpose(sum(np.transpose(correv(np.vstack((zz,ref)),np.vstack((inp[:,:,j],zz)),lenf))))                                        
                                         
            if j == 0:
                Txx = Txxrow
                Vyx = yx
            else:
                Txx = np.vstack((Txx,Txxrow))
                Vyx = np.vstack((Vyx,yx))
    else:
        xx = np.zeros((lenf,nin))
        for j in range(0,nin):
            for i in range(0,nin):
                if i >= j:
                    xx[:,i] = np.transpose(sum(np.transpose(correv(inp[:,:],inp[:,:],lenf))))
                    xx[:,j] = np.transpose(sum(np.transpose(correv(inp[:,:],inp[:,:],lenf))))
                if i == 0:
                    Txxrow = toeplitz(xx[:,i],xx[:,j])
                else:
                    Txxrow = (np.hstack((Txxrow,toeplitz(np.hstack((xx[:,i],xx[:,j]))))))
                    
            yx = np.transpose(sum(np.transpose(correv(np.vstack((zz,ref)),np.vstack((inp[:,:],zz)),lenf))))
                              
            if j == 0:
                Txx = Txxrow
                Vyx = yx
            else:
                Txx = np.vstack((Txx,Txxrow))
                Vyx = np.vstack((Vyx,yx))
            
    # solve the system of equation:
    Tmax = np.max(np.max(Txx))
    Tstab = eps*eps*Tmax*np.eye(nin*lenf)
    ff = np.linalg.solve((Txx+Tstab),Vyx)
    
    # Calculate the output matrix
    out = 0*inp
    f = np.zeros((lenf,nin))
    for j in range (0,nin):
        i1 = (j) * lenf
        i2 = j+1 * lenf
        f[:,j] = ff[i1:i2]
        if len(inp.shape) > 2:
            out[:,:,j] = conv2(inp[:,:,j],f[:,j],mode='same')
        else:
            out[:,:] = conv2(inp[:,:],f[:],mode='same')
        # out[:,:] = convolve2d(inp[:,:],f[:],mode='same')
        
    return out, f
    
             
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import io
# from numpy import linalg as LA
# inp = io.loadmat('/home/hazwanh/Documents/Coding/matlab/demo_clsrme_octave/ls_temp.mat')['P0Ptmp']
# ref = io.loadmat('/home/hazwanh/Documents/Coding/matlab/demo_clsrme_octave/ls_temp.mat')['ref']            
# lenf = io.loadmat('/home/hazwanh/Documents/Coding/matlab/demo_clsrme_octave/ls_temp.mat')['nfilter'][0]
# eps = io.loadmat('/home/hazwanh/Documents/Coding/matlab/demo_clsrme_octave/ls_temp.mat')['eps'][0]  

# eps = eps[0]
# lenf = lenf[0]     
# nfilter = np.ceil(wavc)+1
# nt,nx = inp.shape   
