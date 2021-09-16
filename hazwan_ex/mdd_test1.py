#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:07:51 2021

@author: csi-13

"""

"""
minv_hl,madj_hl,psfinv_hl,psfadj_hl = MDD(Gwav2, d1.T, 
                              dt=par['dt'], dr=par['dx'], nfmax=799, twosided=True, add_negative=False,
                              adjoint=True, psf=True, dtype='complex64', dottest=True, 
                              **dict(damp=1e-10, iter_lim=20, show=1))

def MDD(G, d, dt=0.004, dr=1., nfmax=None, wav=None,
        twosided=True, causality_precond=False, adjoint=False,
        psf=False, dtype='float64', dottest=False,
        saveGt=True, add_negative=True, smooth_precond=0, **kwargs_solver):
"""

G = Gwav2           # shape:(100,60,799)
d = d1.T            # shape:(100,799)
dr = par['dx']      # dr = 2
dt = par['dt']      # dt = 0.004
nfmax = 799
twosided=True
add_negative=False
adjoint=True
psf=True
dtype='complex64'
dottest=True
saveGt=True
causality_precond=False
temp = "**dict(damp=1e-10, iter_lim=20, show=1)"
# **dict(damp=1e-10, iter_lim=20, show=1)
#%%

ncp = get_array_module(d)

ns, nr, nt = G.shape                        # ns=100,nr=60,nt=400
if len(d.shape) == 2:                       # = 2
        nv = 1                              # nv = 1
else:
        nv = d.shape[1]
if twosided:                                # True
   if add_negative:                         # False
            nt2 = 2 * nt - 1
   else:
            nt2 = nt
            nt = (nt2 + 1) // 2             # nt = 400
   nfmax_allowed = int(np.ceil((nt2+1)/2))  # nfmax_allowed = 400
else:
        nt2 = nt
        nfmax_allowed = nt

# Fix nfmax to be at maximum equal to half of the size of fft samples
if nfmax is None or nfmax > nfmax_allowed:
   nfmax = nfmax_allowed                    # nfmax = 400
   logging.warning('nfmax set equal to ceil[(nt+1)/2=%d]' % nfmax)

# Add negative part to data and model
if twosided and add_negative:               # False
   G = np.concatenate((ncp.zeros((ns, nr, nt - 1)), G), axis=-1)
   d = np.concatenate((np.squeeze(np.zeros((ns, nv, nt - 1))), d),
                           axis=-1)
   
# Bring kernel to frequency domain
Gfft = np.fft.rfft(G, nt2, axis=-1)         # (100, 60, 400)
Gfft = Gfft[..., :nfmax]                    # (100, 60, 400)

# Bring frequency/time to first dimension   # Transpose?
Gfft = np.moveaxis(Gfft, -1, 0)             # (400, 100, 60)
d = np.moveaxis(d, -1, 0)                   # from (100, 799) to (799, 100)
if psf:
   G = np.moveaxis(G, -1, 0)                # from (100, 60, 799) to (799, 100, 60)

# Define MDC linear operator
MDCop = MDC(Gfft, nt2, nv=nv, dt=dt, dr=dr, twosided=twosided,
            transpose=False, saveGt=saveGt)             # (79900, 47940) (799x100,799x60)
if psf:
   PSFop = MDC(Gfft, nt2, nv=nr, dt=dt, dr=dr, twosided=twosided,
               transpose=False, saveGt=saveGt)          # (4794000, 2876400) (100x60x799,799x60x60)
   if dottest:
      Dottest(MDCop, nt2*ns*nv, nt2*nr*nv, verb=True,
              backend=get_module_name(ncp))
      if psf:
         Dottest(PSFop, nt2 * ns * nr, nt2 * nr * nr, verb=True,
                 backend=get_module_name(ncp))

# Adjoint
if adjoint:
    madj = MDCop.H * d.flatten()                    # (47940,)
    madj = np.squeeze(madj.reshape(nt2, nr, nv))    # (799,60)
    madj = np.moveaxis(madj, 0, -1)                 # (60,799)
    if psf:
        psfadj = PSFop.H * G.flatten()                      # (2876400,)
        psfadj = np.squeeze(psfadj.reshape(nt2, nr, nr))    # (799,60,60)
        psfadj = np.moveaxis(psfadj, 0, -1)                 # (60,60,799)

# Inverse
if twosided and causality_precond:                  # False
    P = np.ones((nt2, nr, nv))
    P[:nt - 1] = 0
    if smooth_precond > 0:
        P = filtfilt(np.ones(smooth_precond)/smooth_precond, 1, P, axis=0)
    P = to_cupy_conditional(d, P)
    Pop = Diagonal(P)
    minv = PreconditionedInversion(MDCop, Pop, d.flatten(),
                                   returninfo=False, **dict(damp=1e-10, iter_lim=20, show=1))
    # minv = PreconditionedInversion(MDCop, Pop, d.flatten(),
    #                                returninfo=False, **kwargs_solver)
else:
    if ncp == np:                                               # True
        minv = lsqr(MDCop, d.flatten(), **dict(damp=1e-10, iter_lim=20, show=1))[0]     # (47940,)
        # minv = lsqr(MDCop, d.flatten(), **kwargs_solver)[0]
    else:
        minv = cgls(MDCop, d.flatten(), ncp.zeros(int(MDCop.shape[1]),
                                                  dtype=MDCop.dtype),
                    **kwargs_solver)[0]
minv = np.squeeze(minv.reshape(nt2, nr, nv))                    # (799, 60)
minv = np.moveaxis(minv, 0, -1)                                 # (60, 799)

if wav is not None:
    wav1 = wav.copy()
    for _ in range(minv.ndim-1):
        wav1 = wav1[ncp.newaxis]                                # (1, 81)
    minv = get_fftconvolve(d)(minv, wav1, mode='same')          # (60, 799)

if psf:
    if ncp == np:
        psfinv = lsqr(PSFop, G.flatten(), **dict(damp=1e-10, iter_lim=20, show=1))[0]   # (2876400,)
        # psfinv = lsqr(PSFop, G.flatten(), **kwargs_solver)[0]   # (2876400,)
    else:
        psfinv = cgls(PSFop, G.flatten(), ncp.zeros(int(PSFop.shape[1]),
                                                    dtype=PSFop.dtype),
                      **kwargs_solver)[0]
    psfinv = np.squeeze(psfinv.reshape(nt2, nr, nr))            # (799, 60, 60)
    psfinv = np.moveaxis(psfinv, 0, -1)                         # (60, 60, 799)
    if wav is not None:
        wav1 = wav.copy()
        for _ in range(psfinv.ndim-1):
            wav1 = wav1[np.newaxis]                             # (1, 1, 81)
        psfinv = get_fftconvolve(d)(psfinv, wav1, mode='same')  # (799, 60, 60)
        
# Check the results:
    
# true m
plt.figure()
plt.imshow(mwav.T,aspect='auto',interpolation='nearest', 
           vmin=-mwav.max(), vmax=mwav.max(), cmap='gray',
           extent=(x.min(),x.max(),t2.max(),t2.min()))
plt.title('true m', fontsize=15)
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
    
# minv
plt.figure()
plt.imshow(minv.T,aspect='auto',interpolation='nearest', 
           vmin=-minv.max(), vmax=minv.max(), cmap='gray',
           extent=(x.min(),x.max(),t2.max(),t2.min()))
plt.title('minv', fontsize=15)
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
    
# madj
plt.figure()
plt.imshow(madj.T,aspect='auto',interpolation='nearest', 
           vmin=-madj.max(), vmax=madj.max(), cmap='gray',
           extent=(x.min(),x.max(),t2.max(),t2.min()))
plt.title('madj', fontsize=15)
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()