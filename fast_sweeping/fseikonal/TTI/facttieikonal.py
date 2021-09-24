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
        tau[nz, nx]: Numpy array of the traveltime field initialized with infinity everywhere and one at the source
    ''' 
    
    tau = np.ones((nz, nx)) * np.inf
    
    if (0 <= isz <= nz-1) and (0 <= isx <= nx-1):
        tau[isz, isx] = 1.
    else:
        raise ValueError(f"sz and sx must be in the ranges z:({oz}, {oz+(nz-1)*dz}) and x:({ox}, {ox+(nx-1)*dx}), respectively")
    
    return tau

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

def fastsweep_stencil2d(tau, T0, pz0, px0, a, b, c, ip, jp, nz, nx, dz, dx, isz, isx, rhs):
    '''
    ip, jp: indices for the considered point along z and x directions, respectively
    nz, nx: total number of grid points along z and x directions, respectively
    dz, dx: grid spacing along z and x directions, respectively
    T0[nz, nx]: Numpy array of the known traveltime field
    tau[nz, nx]: Numpy array of the unknown traveltime field
    
    Returns: 
    tau[nz, nx]: Numpy array of the traveltime field after updating value at (z,x) = (ip,jp)
    '''
    
    # Return without updating for the source point
    if (ip == isz) and (jp == isx):
        return
    
    # Find the minimum valued neighbor along the z direction
    if (ip == 0) or (ip == nz-1):
        if ip == 0:
            tzmin = tau[1, jp]; T0z = T0[1, jp]; sz = -1
        else:
            tzmin = tau[nz-2, jp]; T0z = T0[nz-2, jp]; sz = 1
    else:
        if (tau[ip-1, jp]*T0[ip-1, jp] < tau[ip+1, jp]*T0[ip+1, jp]):
            tzmin = tau[ip-1, jp]; T0z = T0[ip-1, jp]; sz = 1
        else:
            tzmin = tau[ip+1, jp]; T0z = T0[ip+1, jp]; sz = -1
    
    # Find the minimum valued neighbor along the x direction
    if (jp ==0) or (jp == nx-1):
        if jp == 0:
            txmin = tau[ip, 1]; T0x = T0[ip, 1]; sx = -1
        else:
            txmin = tau[ip, nx-2]; T0x = T0[ip, nx-2]; sx = 1
    else:
        if (tau[ip, jp-1]*T0[ip, jp-1] < tau[ip, jp+1]*T0[ip, jp+1]):
            txmin = tau[ip, jp-1]; T0x = T0[ip, jp-1]; sx = 1
        else:
            txmin = tau[ip, jp+1]; T0x = T0[ip, jp+1]; sx = -1
    
    
    # If both minimum valued neighbors are infinity, return without updating 
    if (txmin == np.inf) and (tzmin == np.inf):
        return
    
    # model values at (ip, jp)
    ap = a[ip, jp]
    bp = b[ip,jp]
    cp = c[ip,jp]; cp2 = cp**2
    T0p = T0[ip,jp]; 
    px0p = px0[ip,jp]
    pz0p = pz0[ip,jp]


    rhsp = rhs[ip,jp]
    
    
    # compute one dimensional traveltime update 
    
    tx = (T0p*txmin + dx*np.sqrt(bp*rhsp/(ap*bp-cp2)))/(T0p + abs(px0p)*dx);
    tz = (T0p*tzmin + dz*np.sqrt(ap*rhsp/(ap*bp-cp2)))/(T0p + abs(pz0p)*dz);

        
    if (tzmin == np.inf): 
        ttemp = tx
    elif (txmin == np.inf):    
        ttemp = tz
    else:
        
        dz2 = dz**2
        dx2 = dx**2
        T0p2 = T0p**2
        
        # Defining additional variables for computational efficiency
        dxpx0p = dx*px0p
        dzpz0p = dz*pz0p
        sxT0p = sx*T0p
        szT0p = sz*T0p
        
        with np.errstate(invalid='ignore'): # ignores warning due to invalid value encountered inside np.sqrt()
        
            # Solving a*dtdx^2 + b*dtdz^2 - 2*c*dtdx*dtdz = rhs (Luo and Qian, 2012)

            root = (dz2*sxT0p*(ap*dxpx0p - cp*dx*pz0p + ap*sxT0p)*txmin \
                 + bp*dx2*T0p2*tzmin - dx*dz*szT0p*(cp*dxpx0p*tzmin \
                 - bp*dx*pz0p*tzmin + cp*sxT0p*(txmin + tzmin)) + dx2*dz2 \
                 *np.sqrt((-2*cp*dx*dz*rhsp*(dxpx0p + sxT0p)*(dzpz0p + szT0p) \
                 + bp*dx2*rhsp*(dzpz0p + szT0p)**2 + cp2*T0p2 \
                 *(dzpz0p*sx*txmin + sx*szT0p*(txmin - tzmin) - dxpx0p*sz*tzmin)**2 \
                 +  ap*(-(T0p2*(dz2*(-rhsp + bp*pz0p**2*txmin**2) \
                 + 2*bp*dzpz0p*szT0p*txmin*(txmin - tzmin) + bp*T0p2 \
                 *(txmin - tzmin)**2)) + 2*dxpx0p*sxT0p*(dz2*rhsp \
                 + bp*dzpz0p*szT0p*txmin*tzmin + bp*T0p2 \
                 *(txmin - tzmin)*tzmin) + dx2*px0p**2*(dz2*rhsp \
                 - bp*T0p2*tzmin**2)))/(dx2*dz2))) /(ap*dz2 \
                 *(dxpx0p + sxT0p)**2 + dx*(dzpz0p + szT0p)*(-2*cp*dz \
                 *(dxpx0p + sxT0p) + bp*dx*(dzpz0p + szT0p)))



        if isCausal2d(root*T0p, ap, bp, cp, tzmin*T0z, txmin*T0x, sz, sx, dz, dx):
            ttemp = root
        else:
            if (tz*T0z < tx*T0x):
                ttemp = tz
            else:
                ttemp = tx
        
    # Update traveltime at (z,x)=(ip,jp) if ttemp is smaller than the previously stored value
    if (ttemp*T0p < tau[ip, jp]*T0p):
        tau[ip, jp] = ttemp        
    
    return

# Function to run the fast sweeping iterations

def fastsweep_run2d(tau, T0, pz0, px0, vz, vx, theta, niter, nz, nx, dz, dx, isz, isx, rhs):
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
                fastsweep_stencil2d(tau, T0, pz0, px0, a, b, c, iz, ix, nz, nx, dz, dx, isz, isx, rhs) 
    
        for iz in range(nz):
            for ix in reversed(range(nx)):
                fastsweep_stencil2d(tau, T0, pz0, px0, a, b, c, iz, ix, nz, nx, dz, dx, isz, isx, rhs) 
                
        for iz in reversed(range(nz)):
            for ix in range(nx):
                fastsweep_stencil2d(tau, T0, pz0, px0, a, b, c, iz, ix, nz, nx, dz, dx, isz, isx, rhs) 
                
        for iz in reversed(range(nz)):
            for ix in reversed(range(nx)):
                fastsweep_stencil2d(tau, T0, pz0, px0, a, b, c, iz, ix, nz, nx, dz, dx, isz, isx, rhs) 
            
    return
