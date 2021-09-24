
import numpy as np

#========== 2D eikonal solver functions ==========


# Function to initialize the traveltime field and check if the source location is within the model

def fastsweep_init2d(nz, nx, dz, dx, isz, isx, oz, ox):
    '''
    nz, nx:   grid numbers along z and x directions, respectively
    dz, dx:   grid spacing along z and x directions, respectively
    isz, isx: z and x grid-coordinates of the source, respectively
    oz, ox:   Origin values along z and x directions, respectively
    
    Returns: 
        T[nz, nx]: Numpy array of the traveltime field initialized with infinity everywhere and zero at the source
    ''' 
    
    T = np.ones((nz, nx)) * np.inf
    
    if (0 <= isz <= nz-1) and (0 <= isx <= nx-1):
        T[isz, isx] = 0.
    else:
        raise ValueError(f"sz and sx must be in the ranges z:({oz}, {oz+(nz-1)*dz}) and x:({ox}, {ox+(nx-1)*dx}), respectively")
    
    return T

def isCausal2d(root, a, b, c, tz, tx, sz, sx, dz, dx):
    dtdz = (root - tz)*sz / dz
    dtdx = (root - tx)*sx / dx
    
    
    vgz_dir = b*dtdz - c*dtdx
    vgx_dir = a*dtdx - c*dtdz
    
    if (vgz_dir*abs(dtdz)*sz > 0.0) and (vgx_dir*abs(dtdx)*sx > 0.0):
        return True
    else:
        return False

# Function to run the inner loop; computes traveltime at each grid point

def fastsweep_stencil2d(T, a, b, c, ip, jp, nz, nx, dz, dx, isz, isx, rhs):
    '''
    ip, jp: indices for the considered point along z and x directions, respectively
    nz, nx: total number of grid points along z and x directions, respectively
    dz, dx: grid spacing along z and x directions, respectively
    T[nz, nx]: Numpy array of the traveltime field
    
    Returns: 
        T[nz, nx] - Numpy array of the traveltime field after updating value at (z,x) = (ip,jp)
    '''
    
    # Return without updating for the source point
    if (ip == isz) and (jp == isx):
        return
    
    # Find the minimum valued neighbor along the z direction
    if (ip == 0) or (ip == nz-1):
        if ip == 0:
            tzmin = T[1, jp]; sz = -1
        else:
            tzmin = T[nz-2, jp]; sz = 1
    else:
        if (T[ip-1, jp] < T[ip+1, jp]):
            tzmin = T[ip-1, jp]; sz = 1
        else:
            tzmin = T[ip+1, jp]; sz = -1
    
    # Find the minimum valued neighbor along the x direction
    if (jp ==0) or (jp == nx-1):
        if jp == 0:
            txmin = T[ip, 1]; sx = -1
        else:
            txmin = T[ip, nx-2]; sx = 1
    else:
        if (T[ip, jp-1] < T[ip, jp+1]):
            txmin = T[ip, jp-1]; sx = 1
        else:
            txmin = T[ip, jp+1]; sx = -1
    
    
    # If both minimum valued neighbors are infinity, return without updating 
    if (txmin == np.inf) and (tzmin == np.inf):
        return
    
    # model values at (ip, jp)
    ap = a[ip, jp]
    bp = b[ip,jp]
    cp = c[ip,jp];cp2 = cp**2

    rhsp = rhs[ip,jp]
    
    # compute one dimensional traveltime update 
    tz = tzmin + dz*np.sqrt(ap*rhsp/(ap*bp-cp2))
    tx = txmin + dx*np.sqrt(bp*rhsp/(ap*bp-cp2))
        
    if (tzmin == np.inf): 
        ttemp = tx
    elif (txmin == np.inf):    
        ttemp = tz
    else:
        
        dz2 = dz**2
        dx2 = dx**2
        

        with np.errstate(invalid='ignore'): # ignores warning due to invalid value encountered inside np.sqrt()
        
            # Solving a*dtdx^2 + b*dtdz^2 - 2*c*dtdx*dtdz = rhs (Luo and Qian, 2012)
            root = (ap*dz2*txmin + dx*dz*np.sqrt(sz*(-2*cp*dx*dz*rhsp*sx + bp*dx2*rhsp*sz \
                 + cp2*sz*(txmin - tzmin)**2) + ap*(dz2*rhsp - bp*(txmin - tzmin)**2)) \
                 + dx*sz*(bp*dx*sz*tzmin - cp*dz*sx*(txmin + tzmin)))/(ap*dz2 + dx*sz*(-2*cp*dz*sx + bp*dx*sz))

        if isCausal2d(root, ap, bp, cp, tzmin, txmin, sz, sx, dz, dx):
            ttemp = root
        else:
            if (tz < tx):
                ttemp = tz
            else:
                ttemp = tx
        
    # Update traveltime at (z,x)=(ip,jp) if ttemp is smaller than the previously stored value
    if (ttemp < T[ip, jp]):
        T[ip, jp] = ttemp        
    
    return

# Function to run the fast sweeping iterations

def fastsweep_run2d(T, vz, vx, theta, niter, nz, nx, dz, dx, isz, isx, rhs):
    '''
    Runs fast sweeping in 2D media.
    Calls fastsweep_stencil2d() to run the inner loop
    ''' 

    a = vx**2*np.cos(theta)**2 + vz**2*np.sin(theta)**2;
    b = vx**2*np.sin(theta)**2 + vz**2*np.cos(theta)**2;
    c = np.sin(theta)*np.cos(theta)*(vz**2-vx**2)
    
    for iloop in range(niter):
        
        for iz in range(nz):
            for ix in range(nx):
                fastsweep_stencil2d(T, a, b, c, iz, ix, nz, nx, dz, dx, isz, isx, rhs) 
    
        for iz in range(nz):
            for ix in reversed(range(nx)):
                fastsweep_stencil2d(T, a, b, c, iz, ix, nz, nx, dz, dx, isz, isx, rhs) 
                
        for iz in reversed(range(nz)):
            for ix in range(nx):
                fastsweep_stencil2d(T, a, b, c, iz, ix, nz, nx, dz, dx, isz, isx, rhs) 
                
        for iz in reversed(range(nz)):
            for ix in reversed(range(nx)):
                fastsweep_stencil2d(T, a, b, c, iz, ix, nz, nx, dz, dx, isz, isx, rhs) 
            
    return
