#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
import pdb, traceback
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def csinci():
    """ Complex valued sinc function interpolation.
    trout = csinci(trin, t, tout, sizetable)
    """

def fktran(D, t, x, ntpad=None, nxpad=None, percent=0., ishift=1):
    """ F-K transform using fft on time domain and ifft on space domain. """
    nsamp = D.shape[0]
    ntr = D.shape[1]

    if len(t) != nsamp:
        raise Exception('Time domain length is inconsistent in input')
    if len(x) != ntr:
        raise Exception('Space domain length is inconsistent in input')

    if ntpad is None:
        ntpad = 2**nextpow2(t)
    if nxpad is None:
        nxpad = 2**nextpow2(x)

    # Get real values of transform with fftrl
    specfx, f = fftrl(D, t, percent, ntpad)

    # Taper and pad in space domain
    if percent > 0.:
        mw = np.tile(mwindow(ntr, percent), (ntr, 1))
        specfx = specfx * mw
    if ntr < nxpad:
        ntr = nxpad                     # this causes ifft to apply the x padding

    spec = np.fft.ifft(specfx.T, n=ntr, axis=0).T
    # Compute kx
    kxnyq = 1. / (2. * (x[1] - x[0]))
    dkx = 2. * kxnyq / ntr
    kx = np.hstack([np.arange(0, kxnyq, dkx), np.arange(-kxnyq, 0, dkx)])

    if ishift:
        tmp = zip(kx, spec)
        tmp.sort()
        kx = [i[0] for i in tmp]
        spec = [i[1] for i in tmp]
    return spec, f, kx


def fftrl(s, t, percent=0.0, n=None):
    """ Returns the real part of the forward Fourier transform. """
    # Determine the number of traces in ensemble
    l = s.shape[0]
    m = s.shape[1]
    ntraces = 1
    itr = 0                             # transpose flag
    if l == 1:
        nsamps = m
        itr = 1
        s = s.T
    elif m == 1:
        nsamps = l
    else:
        nsamps = l
        ntraces = m
    if nsamps != len(t):
        t = t[0] + (t[1] - t[0]) * np.arange(0, nsamps)
    if n is None:
        n = len(t)

    # Apply the taper
    if percent > 0.0:
        mw = np.tile(mwindow(nsamps, percent), (ntraces, 1))
        s = s * mw
    # Pad s if needed
    if nsamps < n:
        s = np.vstack([s, np.zeros([n-nsamps, ntraces])])
        nsamps = n

    # Do the transformation
    spec = np.fft.fft(s, n=nsamps, axis=0)
    spec = spec[:int(n/2)+1, :]              # save only positive frequencies

    # Build the frequency vector
    fnyq = 1. / (2 * (t[1] - t[0]))
    nf = spec.shape[0]
    df = 2.0 * fnyq / n
    f = df * np.arange(0,nf).T
    if itr:
        f = f.T
        spec = spec.T
    return spec, f

def ifktran(spec, f, kx, nfpad=None, nkpad=None, percent=0.0):
    """ Inverse f-k transform.
        Arguments:
            spec    complex valued f-k series
            f       frequency components for rows of spec
            kx      wavenumber components for columns of spec
            nfpad   size to pad spec rows to
            nkpad   size to pad spec columns to
            percent controls cosine taper
        Returns:
            D       2-d array; one trace per column
            t       time coordinates for D
            x       space coordinates for D
    """
    nf,nkx = spec.shape

    if len(f) != nf:
        raise Exception('frequency coordinate vector is wrong size')
    elif len(kx) != nkx:
        raise Exception('wavenumber coordinate vector is wrong size')

    if nfpad is None:
        nfpad = 2**nextpow2(len(f))
    if nkpad is None:
        nkpad = 2**nextpow2(len(kx))

    # Determine if kx needs to be wrapped
    if kx[0] < 0.0:
        # Looks unwrapped (is this wise?)
        ind = kx >= 0.0
        kx = np.hstack([kx[ind], kx[np.arange(ind[0])]])
        spec = np.hstack([spec[:,ind], spec[:,np.arange(ind[0])]])
    else:
        ind = False

    # Taper and pad in kx
    if percent > 0.0:
        mw = mwindow(nkx, percent)
        if ind.any():
            mw = np.hstack([mw[ind], mw[np.arange(ind[0])]])
        mw = mw.repeat(nkz, axis=0)
        spec = spec * mw
    if nkx < nkpad:
        nkx = nkpad

    # Performs the transforms
    specfx = np.fft.fft(spec, nkx)
    D, t = ifftrl(specfx, f)

    # Compute x
    dkx = kx[1] - kx[0]
    xmax = 1.0 / dkx
    dx = xmax / nkx
    x = np.arange(0, xmax, dx)
    return D, t, x

def ifftrl(spec, f):
    """ Inverse Fourier transform for real-valued series.
        Arguments:
            spec    input spectrum
            f       input frequency coordinates
        Returns:
            r       output trace
            t       output time vector
    """
    m,n = spec.shape            # Will be a problem if spec is 1-dimensional
    itr = 0
    if (m == 1) or (n == 1):
        if m == 1:
            spec = spec.T
            itr = 1
        nsamp = len(spec)
        ntr = 1
    else:
        nsamp = m
        ntr = n

    # Form the conjugate symmetric complex spectrum expected by ifft
    # Test for nyquist
    nyq = 0
    if (spec[-1] == np.real(spec[-1])).all():
        nyq = 1
    if nyq:
        L1 = np.arange(nsamp)
        L2 = L1[-2:0:-1]
    else:
        L1 = np.arange(nsamp)
        L2 = L1[-2:0:-1]            # WTF? -njw
    symspec = np.vstack([spec[L1,:], np.conj(spec[L2,:])])
    # Transform the array
    r = (np.fft.ifft(symspec.T)).real.T
    # Build the time vector
    n = len(r)
    df = f[1] - f[0]
    dt = 1.0 / (n*df)
    t = dt * np.arange(n).T
    if itr == 1:
        r = r.T
        t = t.T
    return r, t

def mwindow(n, percent=10.):
    """ 
    w = mwindow(n,percent)
    w = mwindow(n)
    
    MWINDOW returns the N-point Margrave window in a 
    row vector. This window is a boxcar over the central samples
    (100-2*percent)*n/100 in number, while it has a raised cosine
    (hanning style) taper on each end. If n is a vector, it is
    the same as mwindow(length(n)
    
    n= input length of the mwindow. If a vector, length(n) is
       used
    percent= percent taper on the ends of the window
      ************* default=10 ************
    
    by G.F. Margrave, May 1991
    """
    if type(n) is not int and type(n) is not float:
        n = len(n)
    
    # Compute the hanning function
    if percent > 50. or percent < 0.:
        raise Exception('Invalid percent in function mwindow (={0})'.format(percent))
    
    m = 2.0 * math.floor(percent * n / 100.)
    h = np.hanning(m)
    
    return np.hstack([h[:m/2], np.ones([n-m]), h[m/2:]])

def mwhalf(n, percent=10.):
    """ Half mwindow. """
    if type(n) is not int and type(n) is not float:
        n = len(n)
    # Compute the hanning function
    if percent > 100. or percent < 0.:
        raise Exception('Invalid percent in function mwhalf (={0})'.format(percent))
    m = int(math.floor(percent * n / 100.))
    h = np.hanning(2*m)
    return np.hstack([np.ones([n-m]), h[m:0:-1]])

def nextpow2(a):
    """ Gives the next power of 2 larger than a. """
    return np.ceil(np.log(a) / np.log(2)).astype(int)

def fkmig(D, dt, dx, v, params=None):
    """
    Implements an FK (Stolt) migration routine.
  
    Dmig, tmig, xmig = fkmig(D, dt, dx, v, params)
  
        D           data array
        dt          temporal sample rate
        dx          spatial sample rate
        v           constant migration velocity
        params      migration parameters (not yet implemented)
  
    Code translated from CREWES MATLAB algorithm, and similar restrictions
    presumably apply. 
    Taken from Radar Tools library: https://github.com/njwilson23/irlib
    """

    nsamp = D.shape[0]
    ntr = D.shape[1]
    t = np.arange(0, nsamp) * dt
    x = np.arange(0, ntr) * dx
    interpolated = True

    fnyq = 1.0 / (2.0*dt)
    knyq = 1.0 / (2.0*dx)
    tmax = t[-1]
    xmax = abs(x[-1]-x[0])

    # Deal with parameters
    if params == None:
        fmax = 0.6 * fnyq
        fwid = 0.2 * (fnyq - fmax)
        dipmax = 85.0
        dipwid = 90.0 - dipmax
        tpad = min([0.5 * tmax, abs(tmax / math.cos(math.pi*dipmax / 180.0))])
        xpad = min([0.5 * xmax, xmax / math.sin(math.pi*dipmax / 180.0)])
        padflag = 1
        intflag = 3
        cosflag = 1
        lsinc = 1
        ntable = 25
        mcflag = 0      # Faster, less memory-efficient transform (not implemented)
        kpflag = 50.0

    # Apply padding
    # tpad
    nsampnew = int(2.0**nextpow2( round((tmax+tpad) / dt + 1.0) ))
    tmaxnew = (nsampnew-1)*dt
    tnew = np.arange(t[0], tmaxnew+dt, dt)
    ntpad = nsampnew-nsamp
    D = np.vstack([D,np.zeros([ntpad,ntr])])

    # xpad
    ntrnew = 2**nextpow2( round((xmax+xpad) / dx + 1) )
    xmaxnew = (ntrnew-1)*dx + x[0]
    xnew = np.arange(x[0], xmaxnew+dx, dx)
    nxpad = ntrnew - ntr
    D = np.hstack([D, np.zeros([nsampnew,nxpad])])

    # Forward f-k transform
    fkspec, f, kx = fktran(D, tnew, xnew, nsampnew, ntrnew, 0, 0)
    df = f[1] - f[0]
    nf = len(f)

    # Compute frequency mask
    ifmaxmig = int(round((fmax+fwid) / df + 1.0))
    pct = 100.0 * (fwid / (fmax+fwid))
    fmask = np.hstack([mwhalf(ifmaxmig,pct), np.zeros([nf-ifmaxmig])])
    fmaxmig = (ifmaxmig-1)*df       # i.e. fmax+fwid to nearest sample

    # Now loop over wavenumbers
    ve = v / 2.0                    # exploding reflector velocity
    dkz = df / ve
    kz = (np.arange(0,len(f)) * dkz).T
    kz2 = kz ** 2

    th1 = dipmax * math.pi / 180.0
    th2 = (dipmax+dipwid) * math.pi / 180.0
    if th1 == th2:
        print("No dip filtering")

    for j,kxi in enumerate(kx):
        # Evanescent cut-off
        fmin = abs(kxi) * ve
        ifmin = int(math.ceil(fmin / df)) + 1

        # Compute dip mask
        if th1 != th2:
            # First physical frequency excluding dc
            ifbeg = max([ifmin, 1])+1
            # Frequencies to migrate
            ifuse = np.arange(ifbeg, ifmaxmig+1)
            if len(ifuse) == 1:
                # Special case
                dipmask = np.zeros(f.shape)
                dipmask[ifuse-1] = 1
            else:
                # Physical dips for each frequency
                theta = np.arcsin(fmin / f[ifuse])
                # Sample number to begin ramp
                if1 = round(fmin / (math.sin(th1) * df))
                if1 = max([if1, ifbeg])
                # sample number to end ramp
                if2 = round(fmin / (math.sin(th2) * df))
                if2 = max([if2, ifbeg])
                # Initialize mask to zeros
                dipmask = np.zeros(f.shape)
                # Pass these dips
                dipmask[if1:nf-1] = 1
                dipmask[if2:if1] = 0.5 + 0.5 * np.cos(
                        (theta[np.arange(if2, if1, -1) - ifbeg] - th1)
                        * math.pi / float(th2-th1))
        else:
            dipmask = np.ones(f.shape)

        # Apply masks
        tmp = fkspec[:, j] * fmask * dipmask

        # Compute f that map to kz
        fmap = ve * np.sqrt(kx[j]**2 + kz2)
        # Contains one value for each kz giving the frequency
        # that maps there to migrate the data
        # Many of these frequencies will be far too high
        ind = np.vstack(np.nonzero(fmap <= fmaxmig)).T
        # ind is an array of indicies of fmap which will always start at 1
        # and end at the highest f to be migrated

        # Now map samples by interpolation
        fkspec[:, j] *= 0.0             # initialize output spectrum to zero
        if len(ind) != 0:
            # Compute cosine scale factor
            if cosflag:
                if fmap[ind].all() == 0:
                    scl = np.ones(ind.shape[0])
                    li = ind.shape[0]
                    scl[1:li] = (ve * kz[ind[1:li]] / fmap[ind[1:li]])[:,0]
                else:
                    scl = ve * kz[ind] / fmap[ind]
            else:
                scl = np.ones(ind.shape[0])
            if intflag == 0:
                # Nearest neighbour interpolation
                ifmap = (fmap[ind] / df).astype(int)
                fkspec[ind, j] = (scl.squeeze() \
                    * tmp[ifmap.squeeze()]).reshape([-1,1])
            elif intflag == 1:
                # Complex sinc interpolation
                fkspec[ind, j] = scl \
                        * csinci(tmp, f, fmap[ind], np.hstack([lsinc,ntable]))
            elif intflag == 2:
                # Spline interpolation
                # Not implemented
                pass
            elif intflag == 3:
                # Linear interpolation
                r_interp = scl.squeeze() \
                    * np.interp(fmap[ind], f, tmp.real).squeeze()
                j_interp = scl.squeeze() \
                    * np.interp(fmap[ind], f, tmp.imag).squeeze()
                fkspec[ind, j] = (r_interp + j_interp * 1j).reshape([-1,1])

    # Inverse transform
    Dmig, tmig, xmig = ifktran(fkspec, f, kx)

    # Remove padding, if desired
    if padflag:
        Dmig = Dmig[:nsamp, :ntr]
        tmig = tmig[:nsamp]
        xmig = xmig[:ntr]

    return Dmig, tmig, xmig

def convz(r,w,nzero=None,nout=None,flag=0):
    """
    
    function is designed for a convenient convolution of a seismic
    trace with a zero phase (no time delay) wavelet. This can 
    actually be used with any non-causal wavelet by specifying the
    nzero sample which is the sample number of zero time. It
    defaults to the middle of w but can be placed anywhere. Also, 
    this is designed to produce an output vector of length equal
    to the first input vector (r). Uses MATLAB's CONV function.
    
    s= output trace of length nout
    r= input trace (reflectivity)
    w= input wavelet
    nzero= sample number of zero time value for wavelet
     *********** default=round((length(wavelet)+1)/2)
    nout= length of output trace. 
      ********** default=length(r)
    flag= 1 --> apply a cosine taper at the beginning and end of the
                output trace
       = 0 --> don't apply a taper
         ********* default= 0 **********
    
    by G.F. Margrave, May 1991
    
    @author: hazwanh
    """
    from scipy import signal
    
    if type(nout) == type(None):
        nout = len(r)
    if type(nzero) == type(None):
        nzero = int(np.round(len(w)/2))
        
    # convert r and w to array
    r = np.array([r])
    w = np.array([w])
        
    # convert to column vector
    a,b = r.shape

    if a == 1:
        r = r.T
    w = w.T
    
    temp = signal.convolve(r,w)
    s = temp[nzero-1:nout+nzero-1]
    
    if flag == 1:
        s = s*((mwindow(nout,4)).T)
        
    if a == 1:
        s = s.T
        
    return s
        
def ricker(dt,fdom=None,tlength=None):
    """

    RICKER returns a Ricker wavelet.
    
    dt= desired temporal sample rate
    fdom= dominant frequency in Hz (default: 15 Hz)
    tlength= wavelet length in seconds (default: 127*dt 
                                        (ie a power of 2))
     
    The wavelet is generated from an analog expression for a 
    Ricker wavelet.
     
    by G.F. Margrave, May 1991

    """
    
    import numpy as np
    
    if type(fdom) == type(None):
        fdom = 15
        
    if type(tlength) == type(None):
        tlength=127.*dt
        
    # create a time vector
    nt = np.round(tlength/dt)+1
    tmin = -dt*np.round(nt/2)
    tmax = -tmin-dt
    tw = np.arange(tmin,tmax+dt,dt)
    
    # create the wavelet
    pf = np.pi**2*fdom**2
    wavelet = (1-2*pf*tw**2)*np.exp(-pf*tw**2)
    
    # normalize
    # generate a reference sinusoid at the dominant frequency
    refwave = np.sin(2*np.pi*fdom*tw)
    reftest = convz(refwave,wavelet)
    
    fact = np.max(refwave)/np.max(reftest)
    wavelet = wavelet*fact
    
    return wavelet,tw

def fdacmod(vgrid, dx, dtwav, wav, dipmon, xsrc, zsrc, xrcv1, xrcv2, dxrcv, 
            zrcv1, zrcv2, dzrcv, dtrcv, nxextra, nzextra):
    """
    Created on Fri Oct  8 22:45:47 2021
    
    Generate the synthetic data using the acoustic finite-difference approach
    
    The required inputs:
        
        vgrid       : uniformly gridded velocity model, assume origin at (0,0)
        dx          : sampling in velocity model (dz=dx)
        dtwav       : time sampling of source wavelet and modeling step
        wav         : 1D signal with source wavelet, length defines total modeling time
        dipmon      : 'd' is dipole, 'm' is monopole source
        xsrc        : 1D array with source x-locations
        zsrc        : 1D array with source z-locations
        xrcv1       : first receiver x-location
        xrcv2       : last receiver x-location
        drxcv       : increment in receiver x-location
                      if dxsrc=0: vertical receivers             
        zrcv1       : first receiver z-location
        zrcv2       : last receiver z-location
        drzcv       : increment in receiver z-location
                      if dzsrc=0: horizontal receivers
        dtrcv       : sampling of recorded data (integer multiple of dtwav)
        nxextra     : extend model to left and right with nxextra gridpoints
        nzextra     : extend model in vertical direction with nzextra gridpoints
   
    The outputs:
        
        shot        : 3D array with recorded shot records (ntrcv*nrcv*nsrc)
        
    Global parameter
        
       global verbose_fdacmod 
    
    Examples:
        
    shot = fdacmod(vgrid,dx,dtwav,wav,dipmon,xsrc,zsrc,xrcv1,xrcv2,dxrcv,
                   zrcv1,zrcv2,dzrcv,dtrcv,nxextra,nzextra)
    
    ##########################################################################
    
    Notes:
    1) There are circular boundary conditions: 
       - wavefield at the bottom come back at the top and v.v.
       - wavefields that go out at the left come back at the right and v.v.
    2) The sampling of the source wavelet defines the modeling time step
    3) The length of the source wavelet defines the modeling time
    4) Source/receiver locations are always rounded to grid locations
        
    Author: Eric Verschuur
            Delft University of Technology
    Date  : 8 November 2011
    Update: 20 February 2015
    """
    global verbose_fdacmod
    
    # extract size of the model
    nz, nx = vgrid.shape
    dz = dx
    
    # define some axes for display
    xmax = (nx-1)*dz
    zmax = (nz-1)*dz
    x = np.array([np.linspace(0,xmax,nx)])
    z = np.array([np.linspace(0,zmax,nz)])
    
    # retrieve number of sources from array
    nsrc = np.max(xsrc.shape)
    if np.max(xsrc.size) != np.max(zsrc.size):
        print('Error: number of xsrc and zsrc must be equal')
    else:
        print(f'Found {nsrc} sources')
        
    if (dxrcv != 0) and (dzrcv != 0):
        print('Program cannot handle both dxrcv and dzrcv not being zero!!')
        
    incdxr = int(np.round(dxrcv/dx))
    incdzr = int(np.round(dzrcv/dz))
    
    if dxrcv == 0:
        xrcv = xrcv1
    else:
        xrcv = np.arange(xrcv1,xrcv2+dxrcv,dxrcv)
        nrcv = xrcv.shape
        
    if dzrcv == 0:
        zrcv = zrcv1
    else:
        zrcv = np.arange(zrcv1,zrcv2+dzrcv,dzrcv)
        nrcv = zrcv.shape
        
    # Extract length of wavelet
    ntmod = np.max(wav.shape)
        
    # Calculate some min/max values:
    # assume fmax = 2*fdom
    df = 1/(ntmod*dtwav)
    wavmax = np.max(np.round(np.abs(np.fft.fft(wav)),8))
    ifmax = np.argmax(np.round(np.abs(np.fft.fft(wav)),8),axis=0)
    fmax = 2*(ifmax-1)*df
    vmin = np.min(vgrid)
    vmax = np.max(vgrid)
    dtwavmax = dx/(np.sqrt(2)*vmax)

    # Display information on the modelling
    print(f'Minimum velocity in model is {vmin}')
    print(f'Maximum frequency in wavelet is approximately {fmax}')
    print(f'Dx should be smaller than {vmin/10*fmax}')
    print(f'Dx of the model is {dx}')
    print(f'Maximum velocity in model is {vmax:.2f}')
    print(f'Maximum allowed sampling is {dtwavmax:.4f}')
    print(f'Sampling wavelet is {dtwav}')
    
    if dtwav > dtwavmax:
        print(f'Error: wavelet sampling is too coarse')
        
    if dipmon =='d':
        print('Dipole source')
    else:
        if dipmon == 'm':
            print('Monopole source')
        else:
            print('Error: choose dipmon = d or m')
    
    #--------------------------------------------------------------------------
    # Extend model over nxextra and nzextra gridpoints
    #--------------------------------------------------------------------------
    
    nxext = nx+2*nxextra
    nzext = nz+nzextra
    xextra = nxextra*dx
    vmin = vgrid[0,0]
    
    vgridext = vmin+np.zeros((nzext,nxext))
    vgridext[0:nz,nxextra:nxextra+nx] = vgrid
    
    # Fade out the sides over nxextra
    for ix in range(0,nxextra):
        fact = (ix)/(nxextra - 1)
        cosfact = (np.cos(fact*np.pi/2))**2
        vgridext[0:nz,ix] = fact*vgrid[:,0]+(1-fact)*vmin
        vgridext[0:nz,nxext-ix-1] = fact*vgrid[:,nx-1]+(1-fact)*vmin
        
    # Fade out the model over the lower side
    for ix in range(0,nxext):
        fact = 0
        for iz in range(0,nzextra):
            fact = (iz)/(nzextra-1)
            cosfact = (np.cos(fact*np.pi/2))**2
            vgridext[nz+iz,ix] = (1-cosfact)*vmin+cosfact*vgridext[nz-1,ix]
    
    if verbose_fdacmod > 0:
        fig, ax = plt.subplots()
        im = plt.imshow(vgridext,cmap='rainbow')
        ax.set_title('fdacmod: extended velocity model')
        plt.colorbar(im)
        plt.axis('tight')
        
    #--------------------------------------------------------------------------
    # Determine time step increment for receivers
    #--------------------------------------------------------------------------
    
    incdt = np.round(dtrcv/dtwav)
    ntrcv = int(1 + np.round(ntmod/incdt))
    trcv = np.arange(0,(ntrcv-1)*incdt*dtwav+(incdt*dtwav),incdt*dtwav)
    
    # Make array 'shots' for shot records
    shots = np.zeros((ntrcv,nrcv,nsrc))
    
    #--------------------------------------------------------------------------
    # Run FD modelling for each source
    #--------------------------------------------------------------------------
    
    for isrc in range(0,nsrc):
        
        ixsrc = int(nxextra + 1 + np.round(xsrc[isrc]/dx))
        izsrc = int(1 + np.round(zsrc[isrc]/dz))
        
        izsrcmin1 = int(izsrc-1)
        if izsrcmin1 <= 0:
            izsrcmin1 = int(izsrcmin1 + nzext)
        
        izsrcplus1 = int(izsrc+1)
        if izsrcmin1 > nzext:
            izsrcplus1 = int(izsrcplus1 - nzext)
            
        ixrcv1 = nxextra + 1 + np.round(xrcv1/dx)
        ixrcv2 = nxextra + 1 + np.round(xrcv2/dx)
        izrcv1 = 1 + np.round(zrcv1/dx)
        izrcv2 = 1 + np.round(zrcv2/dx)
        
        print(f'Modeling source (x,z) = {xsrc[isrc]},{zsrc[isrc]}')
    
        # Time step: 0; also prepare for output shot
        
        p1 = np.zeros((nzext,nxext))
        p2 = np.zeros((nzext,nxext));
        factor = dtwav*dtwav*vgridext*vgridext/(dx*dx);
        
        if dipmon == 'd':
            p2[izsrcmin1-1,ixsrc-1] = np.float(-dtwav*dtwav*wav[0])
            p2[izsrcplus1-1,ixsrc-1] = np.float(dtwav*dtwav*wav[0])
        else:
            p2[izsrc-1,ixsrc-1] = np.float(dtwav*dtwav*wav[0])
            
        itrcv = 1;
        itloc = 1;    
        
        # Two steps: from p2 to p1 and from p1 to p2
        
        for it in np.arange(2,ntmod,2):
            dum = np.roll(p2,1,axis=0) + np.roll(p2,-1,axis=0)+ np.roll(
                np.roll(p2,1,axis=1),0,axis=0) + np.roll(np.roll(p2,-1,axis=1),0,axis=0)-4*p2
            p1 = -p1+2*p2+factor*dum
            
            if dipmon == 'd':
                p1[izsrcmin1-1,ixsrc-1] = p1[izsrcmin1-1,ixsrc-1]-dtwav*dtwav*wav[it-1]
                p1[izsrcplus1-1,ixsrc-1] = p1[izsrcplus1-1,ixsrc-1]+dtwav*dtwav*wav[it-1]
            else:
                p1[izsrc-1,ixsrc-1] = p1[izsrc-1, ixsrc-1] + dtwav*dtwav*wav[it-1]
                
            # Save in receiver file for the right sample
            
            if np.remainder(itloc,incdt) == 0:
                if incdxr == 0:
                   # shots[itrcv-1,:,isrc] = p1[izrcv1-1:izrcv2+incdzr:incdzr,ixrcv1-1]
                   shots[itrcv-1,:,isrc] = p1[int(izrcv1-1):int(izrcv2):int(incdzr),int(ixrcv1-1)]
                else:
                   # shots[itrcv-1,:,isrc] = p1[izrcv1-1,ixrcv1-1:ixrcv2+incdxr:incdxr]
                   shots[itrcv-1,:,isrc] = p1[int(izrcv1-1),int(ixrcv1-1):int(ixrcv2):int(incdxr)]
                itrcv = itrcv + 1
    
            itloc = itloc + 1
            
            dum = np.roll(p1,1,axis=0) + np.roll(p1,-1,axis=0)+ np.roll(
                np.roll(p1,1,axis=1),0,axis=0) + np.roll(np.roll(p1,-1,axis=1),0,axis=0)-4*p1;
            p2 = -p2 + 2 * p1 + factor * dum
            
            if dipmon == 'd':
                p2[izsrcmin1-1,ixsrc-1] = p2[izsrcmin1-1,ixsrc-1]-dtwav*dtwav*wav[it]
                p2[izsrcplus1-1,ixsrc-1] = p2[izsrcplus1-1,ixsrc-1]+dtwav*dtwav*wav[it]
            else:
                p2[izsrc-1,ixsrc-1] = p2[izsrc-1, ixsrc-1] + dtwav*dtwav*wav[it]
                
            # Save in receiver file for the right sample
            if np.remainder(itloc,incdt) == 0:
                if incdxr == 0:
                   # shots[itrcv-1,:,isrc] = p1[izrcv1-1:izrcv2+incdzr:incdzr,ixrcv1-1]
                   shots[itrcv-1,:,isrc] = p2[int(izrcv1-1):int(izrcv2):int(incdzr),int(ixrcv1-1)]
                else:
                   # shots[itrcv-1,:,isrc] = p1[izrcv1-1,ixrcv1-1:ixrcv2+incdxr:incdxr]
                   shots[itrcv-1,:,isrc] = p2[int(izrcv1-1),int(ixrcv1-1):int(ixrcv2):int(incdxr)]
                itrcv = itrcv + 1
                
            itloc = itrcv + 1
            
            # Give message and show snap shot every verbose_fdacmod time steps
            
            if verbose_fdacmod > 0:
                if np.remainder(it,verbose_fdacmod) == 0:
                    print(f'Time step is {it*dtwav}')
                    fig, ax = plt.subplots()
                    im = plt.imshow(p2[0:nz,nxextra:nxextra+nx+1],cmap='gray')
                    plt.axis('tight')
                    plt.show()
                    plt.close()
            # end loops after time steps
                
        # Display current shot record
        
        if verbose_fdacmod > 0:
            fig, ax = plt.subplots()
            if incdxr == 0:
                im = plt.imshow(shots[:,:,isrc], vmin=0.2*np.min(shots[:,:,isrc]),
                                vmax=0.2*np.max(shots[:,:,isrc]) ,cmap='gray')
            else:
                im = plt.imshow(shots[:,:,isrc],vmin=0.2*np.min(shots[:,:,isrc]),
                                vmax=0.2*np.max(shots[:,:,isrc]), cmap='gray')
            plt.axis('tight')
            ax.set_title(f'fdacmod: modeled response shot no:{isrc}')
            plt.axis('tight')
            
        return shots

def make_mask(nt,nx,triangle,top):
    
    import numpy as np
    """
	Function: make_mask

	Description: Creates a simple triangular mask with a flattened mute and the top. 
                 So it is basically a trapezoid.

	Arguments:

		Nt::Int,
		=> Length of trace in time

		Nx::Int,
		=> Number of traces ie. number of receivers/sources

		triangle::Array{Int,2},
		=> List of 2D positions of corners of the triangle

		top::Int
		=> Position of top

	Return:

		Mask::Array{Float64,2}
		=> Output mask

    """
    
    # read in triangle point
    xa = triangle[0,0]
    xb = triangle[1,0]
    xc = triangle[2,0]
    
    ta = triangle[0,1]
    tb = triangle[1,1]
    tc = triangle[2,1]
    
    # Mask = np.ones(nt,2*nx)
    Mask = np.ones((nt,nx))
    
    # top
    Mask[0:top,:] = 0.0
    
    # sides
    for x in range(xa,xb+1):
        t = np.round(ta+(tb-ta)/(xb-xa)*(x-xa))
        Mask[0:int(t),x] = 0.0
        
    for x in range(xb,xc+1):
        t = np.round(tb+(tc-tb)/(xc-xb)*(x-xb))
        Mask[0:int(t),x] = 0.0
        
    return Mask

def padding(data, padsize, padval=0, direction='post'):
    """
    
    padding : This functions pads with zeros a given array
    
        out = padding(in, padsize, padval, direction)
    
    IN
        in : is the input array
        padsize : is the amount of padding to be done
        padval : is the value to be padded with
        direction : is the direction to pad along
                  = 'post' : pad at the end
                  = 'pre' :  pad at the begining
     
    OUT
        out : padded array
    
    EXAMPLE
        A = np.random.rand(5,1)
        B = padding(A,10,1,'post');
    
    """
    
    if direction == 'post':
        out = np.vstack((data,padval*np.ones((padsize,1))))
    elif direction == 'pre':
        out = np.vstack((padval*np.ones((padsize,1)),data))
    else:
        raise ValueError('Please input a valid direction parameter. See help.')
        
    return out

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

# nwav = 15;
# if isrc in range(0,nsrc):
#     # find A using lsqr filter
#     nfilter = nwav
#     A_tmp = leastsub(PPin[:,:,isrc],P[:,:,isrc],nfilter,eps)[1]
#     A[:,isrc] = np.roll(np.roll(padding(A_tmp,np.round(pad*nt-nfilter,0,'post')),
#                                 -np.round(nfilter/2)+1,axis=0),0,axis=0)
    