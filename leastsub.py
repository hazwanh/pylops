#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:40:19 2021

@author: csi-13
"""
def correv(a, b, l):
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
    aa[1:nta,:] = a
    bb[1:ntb,:] = b
    for j in range(1,l+1):
        c[j,:]  = sum(aa[j:nt,:]*bb[1:nt+1-j,:])



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
    from
    
    nt,nx,nin = size(inp)
    ntr,nxr = size(ref)
    
    nrot = np.floor((lenf-1)/2)
    zz = np.zeros((nrot,nx))
    
    xx = np.zeros(lenf,nin,nin)
    for j in range(1,nin+1):
        j
        for i in range(1,nin+1):
            if i >= 1:
                xx[:,i,j] = np.sum(correv(inp[:,:,i],inp[:,:,j],lenf))
                xx[:,j,i] = np.sum(correv(inp[:,:,j],inp[:,:,i],lenf))
            if i == 1:
                Txxrow = toeplitz(xx,[:,i,j],xx[:,j,i])
            elif:
                Txxrow = [Txxrow,toeplitz(xx[:,i,j],xx[:,j,i])]
                
        yx = np.sum(correv([zz;ref],[inp([:,:,j];zz],lenf)))
        if j == 1:
            Txx = Txxrow
            Vyx = yx
        elif:
            Txx = [Txx;Txxrow]
            Vyx = [Vyx;yx]
            
    # solve the system of equation:
    Tmax = max(max(Txx))
    Tstab = eps*eps*Tmax*np.eye(nin*lenf)
    ff = (Txx+Tstab)\Vyx
    
    # Calculate the output matrix
    out = 0*inp
    f = np.zeros((lenf,nin))
    for j in range (1,nin+1):
        i1 = (j-1)*lenf+1
        i2 = j*lenf
        f[:,j] = ff[i1:i2]
        out[:,:,j] = conv2(inp[:,:,j],f[:,j],mode='same')
        
    return out f
    
                    
                
            