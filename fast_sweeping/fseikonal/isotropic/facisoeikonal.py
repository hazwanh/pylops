import numpy as np


#========== 2D eikonal solver functions ==========

# Function to initialize the traveltime field and check if the source location is within the model
def fastsweep_init2d(nz, nx, dz, dx, isz, isx, oz, ox):
    '''
    Inputs:
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

# Function to run the inner loop; computes traveltime at each grid point
def fastsweep_stencil2d(tau, v, T0, pz0, px0, ip, jp, nz, nx, dz, dx, isz, isx):
    '''
    Inputs:
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
            tzmin = tau[1, jp]; T0z = T0[1, jp]; 
        else:
            tzmin = tau[nz-2, jp]; T0z = T0[nz-2, jp]; 
    else:
        if (tau[ip-1, jp]*T0[ip-1, jp] < tau[ip+1, jp]*T0[ip+1, jp]):
            tzmin = tau[ip-1, jp]; T0z = T0[ip-1, jp]; 
        else:
            tzmin = tau[ip+1, jp]; T0z = T0[ip+1, jp]; 
    
    # Find the minimum valued neighbor along the x direction
    if (jp ==0) or (jp == nx-1):
        if jp == 0:
            txmin = tau[ip, 1]; T0x = T0[ip, 1]; 
        else:
            txmin = tau[ip, nx-2]; T0x = T0[ip, nx-2]; 
    else:
        if (tau[ip, jp-1]*T0[ip, jp-1] < tau[ip, jp+1]*T0[ip, jp+1]):
            txmin = tau[ip, jp-1]; T0x = T0[ip, jp-1]; 
        else:
            txmin = tau[ip, jp+1]; T0x = T0[ip, jp+1]; 
    
    # If both minimum valued neighbors are infinity, return without updating 
    if (txmin == np.inf) and (tzmin == np.inf):
        return

    
    # velocity value at (ip, jp)
    vp = v[ip, jp]

    T0p = T0[ip,jp]
    px0p = abs(px0[ip,jp])*np.sign(T0p-T0x)
    pz0p = abs(pz0[ip,jp])*np.sign(T0p-T0z)

    # compute one dimensional traveltime update 

    tx = (dx + T0p*txmin*vp)/(dx*px0p*vp + T0p*vp)
    tz = (dz + T0p*tzmin*vp)/(dz*pz0p*vp + T0p*vp)
        
    if (tz*T0p <= txmin*T0x): 
        ttemp = tz

    elif (tx*T0p <= tzmin*T0z):    
        ttemp = tx

    else:
        
        dx2 = dx*dx
        dz2 = dz*dz
        T0p2 = T0p**2
        dx2dz2 = dx2*dz2

        ttemp = (dx*dz2*px0p*T0p*txmin + dz2*T0p2*txmin + dx2*T0p*(dz*pz0p + T0p)*tzmin \
              + dx2dz2*np.sqrt((dx2dz2*px0p**2 + dx2dz2*pz0p**2 + 2*dx*dz**2*px0p*T0p \
              + 2*dx2*dz*pz0p*T0p + dx2*T0p2 + dz2*T0p2 - T0p2*(dz*pz0p*txmin \
              + T0p*txmin - (dx*px0p + T0p)*tzmin)**2*vp**2)/(dx2dz2*vp**2)))/(dx2dz2 \
              *(px0p**2 + pz0p**2) + 2*dx*dz*(dz*px0p + dx*pz0p)*T0p + (dx2 + dz2)*T0p2)



    # Update traveltime at (z,x)=(ip,jp) if ttemp*T0p is smaller than the previously stored value
    if (ttemp*T0p < tau[ip, jp]*T0p):
        tau[ip, jp] = ttemp        
    
    return

# Function to run the fast sweeping iterations
def fastsweep_run2d(tau, v, T0, pz0, px0, niter, nz, nx, dz, dx, isz, isx):
    '''
    Runs fast sweeping in 2D media.
    Calls fastsweep_stencil2d() to run the inner loop
    ''' 

    for iloop in range(niter):
        
        for iz in range(nz):
            for ix in range(nx):
                fastsweep_stencil2d(tau, v, T0, pz0, px0, iz, ix, nz, nx, dz, dx, isz, isx) 
    
        for iz in range(nz):
            for ix in reversed(range(nx)):
                fastsweep_stencil2d(tau, v, T0, pz0, px0, iz, ix, nz, nx, dz, dx, isz, isx) 
                
        for iz in reversed(range(nz)):
            for ix in range(nx):
                fastsweep_stencil2d(tau, v, T0, pz0, px0, iz, ix, nz, nx, dz, dx, isz, isx) 
                
        for iz in reversed(range(nz)):
            for ix in reversed(range(nx)):
                fastsweep_stencil2d(tau, v, T0, pz0, px0, iz, ix, nz, nx, dz, dx, isz, isx) 
            
    return


#========== 3D eikonal solver functions ==========

# Function to initialize the traveltime field and check if the source location is within the model
def fastsweep_init3d(ny, nz, nx, dy, dz, dx, isy, isz, isx, ymin, zmin, xmin):
    '''
    Inputs:
        ny, nz, nx:   grid numbers along y, z, and x directions, respectively
        dy, dz, dx:   grid spacing along y, z, and x directions, respectively
        isy, isz, isx: y, z, and x grid-coordinates of the source, respectively
        oy, oz, ox:   Origin values along y, z, and x directions, respectively
    
    Returns: 
        tau[ny, nz, nx]: Numpy array of the traveltime field initialized with infinity everywhere and one at the source
    '''
    
    tau = np.ones((ny, nz, nx)) * np.inf
    
    if (0 <= isy <= ny-1) and (0 <= isz <= nz-1) and (0 <= isx <= nx-1):
        tau[isy, isz, isx] = 1.
    else:
        raise ValueError(f"sy, sz, and sx must be in the ranges \n y:({oy}, {oy+(ny-1)*dy}), \n z:({oz}, {oz+(nz-1)*dz}), and x:({ox}, {ox+(nx-1)*dx}), respectively")
    
    return tau
    
# Function to run the inner loop; computes traveltime at each grid point
def fastsweep_stencil3d(tau, v, T0, py0, pz0, px0, kp, ip, jp, ny, nz, nx, dy, dz, dx, isy, isz, isx):
    '''
    Inputs:
        kp, ip, jp: indices for the considered point along y, z, and x directions, respectively
        ny, nz, nx: total number of grid points along y, z, and x directions, respectively
        dy, dz, dx: grid spacing along y, z, and x directions, respectively
        T0[ny, nz, nx]: Numpy array of the known traveltime field
        tau[ny, nz, nx]: Numpy array of the unknown traveltime field
    
    Returns: 
        tau[ny, nz, nx]: Numpy array of the unknown traveltime field after updating value at (y,z,x) = (kp,ip,jp)
    '''
    
    if (kp == isy) and (ip == isz) and (jp == isx):
        return

    # Find the minimim valued neighbor along y direction 

    if (kp == 0) or (kp == ny-1):
        if kp == 0:
            tymin = tau[1,ip,jp]; T0y = T0[1,ip,jp]; sy=-1;
        else:
            tymin = tau[ny-2,ip,jp]; T0y = T0[ny-2,ip,jp]; sy=1
    else:
        if (tau[kp-1,ip,jp]*T0[kp-1,ip,jp] < tau[kp+1,ip,jp]*T0[kp+1,ip,jp]):
            tymin = tau[kp-1,ip,jp]; T0y = T0[kp-1,ip,jp]; sy=1
        else:
            tymin = tau[kp+1,ip,jp]; T0y = T0[kp+1,ip,jp]; sy=-1 
    
    # Find the minimim valued neighbor along z direction

    if (ip == 0) or (ip == nz-1):
        if ip == 0:
            tzmin = tau[kp,1,jp]; T0z = T0[kp,1,jp]; sz=-1
        else:
            tzmin = tau[kp,nz-2,jp]; T0z = T0[kp,nz-2,jp]; sz=1
    else:
        if (tau[kp,ip-1,jp]*T0[kp,ip-1,jp] < tau[kp,ip+1,jp]*T0[kp,ip+1,jp]):
            tzmin = tau[kp,ip-1,jp]; T0z = T0[kp,ip-1,jp];sz=1
        else:
            tzmin = tau[kp,ip+1,jp]; T0z = T0[kp,ip+1,jp]; sz=-1
    
    # Find the minimim valued neighbor along x direction   

    if (jp == 0) or (jp == nx-1):
        if jp == 0:
            txmin = tau[kp,ip,1]; T0x = T0[kp,ip,1]; sx=-1
        else:
            txmin = tau[kp,ip,nx-2]; T0x = T0[kp,ip,nx-2]; sx=1
    else:
        if (tau[kp,ip,jp-1]*T0[kp,ip,jp-1] < tau[kp,ip,jp+1]*T0[kp,ip,jp+1]):
            txmin = tau[kp,ip,jp-1]; T0x = T0[kp,ip,jp-1]; sx=1
        else:
            txmin = tau[kp,ip,jp+1]; T0x = T0[kp,ip,jp+1]; sx=-1
             
    
    if (tymin == np.inf) and (txmin == np.inf) and (tzmin == np.inf):
        return

    # Velocity value at the considered point

    vp = v[kp,ip,jp]
    T0p = T0[kp,ip,jp]; T0p2 = T0p**2
    py0p = abs(py0[kp,ip,jp])*np.sign(T0p-T0y); py0p2 = py0p**2
    pz0p = abs(pz0[kp,ip,jp])*np.sign(T0p-T0z); pz0p2 = pz0p**2
    px0p = abs(px0[kp,ip,jp])*np.sign(T0p-T0x); px0p2 = px0p**2


    td = [(tymin*T0y,tymin,dy,T0y,py0p),(tzmin*T0z,tzmin,dz,T0z,pz0p),(txmin*T0x,txmin,dx,T0x,px0p)]
    td.sort(key=lambda x: x[0])

    tmin = [i[1] for i in td]
    d = [i[2] for i in td]
    T0min = [i[3] for i in td]
    p0 = [i[4] for i in td]

    ttemp = (d[0] + T0p*tmin[0]*vp)/(d[0]*abs(p0[0])*vp + T0p*vp) 

    if (ttemp*T0p)>(tmin[1]*T0min[1]):

        d02 = d[0]**2; d04 = d02**2
        d12 = d[1]**2; d14 = d12**2

        with np.errstate(invalid='ignore'): # ignores warning due to invalid value encountered inside np.sqrt()

            ttemp = (T0p*d12*(T0p + d[0]*p0[0])*tmin[0] + T0p2*d02*tmin[1] \
                  + T0p*d02*d[1]*p0[1]*tmin[1] + (d02*d12 *np.sqrt((4*(T0p*d12 \
                  *(T0p + d[0]*p0[0])*tmin[0] + T0p*d02 *(T0p + d[1]*p0[1]) \
                  *tmin[1])**2)/(d04*d14) - 4*(T0p2*(d[0]**(-2) + d[1]**(-2)) \
                  + p0[0]**2 + p0[1]**2 + 2*T0p*(p0[0]/d[0] + p0[1]/d[1])) \
                  *(-vp**(-2) + T0p2 *(tmin[0]**2/d02 + tmin[1]**2/d12))))/2.) \
                  /(T0p2 *(d02 + d12) + 2*T0p*d[0]*d[1]*(d[1]*p0[0] + d[0]*p0[1]) \
                  + d02*d12*(p0[0]**2 + p0[1]**2))

        if (ttemp*T0p)>(tmin[2]*T0min[2]):

            dy2 = dy**2
            dz2 = dz**2
            dx2 = dx**2

            with np.errstate(invalid='ignore'): # ignores warning due to invalid value encountered inside np.sqrt()
                ttemp = (dx*dy2*dz2*px0p*T0p*txmin + dy2*dz2*T0p2*txmin + dx2*T0p \
                      *(dz2*(dy*py0p + T0p)*tymin + dy2*(dz*pz0p + T0p) *tzmin) \
                      + (dx2*dy2*dz2*np.sqrt(4*T0p2*((px0p*txmin)/dx + (T0p*txmin)/dx2 \
                      + ((dy*py0p + T0p)*tymin)/dy2 + ((dz*pz0p + T0p)*tzmin)/dz2)**2 \
                      - 4*(px0p2 + py0p2 + pz0p2 + (2*px0p*T0p)/dx + (2*py0p*T0p)/dy \
                      + T0p*((dx**(-2) + dy**(-2)) *T0p + (2*dz*pz0p + T0p)/dz2)) \
                      *(T0p2*(txmin**2/dx2 + tymin**2/dy2 + tzmin**2/dz2) - vp**(-2))))/2.) \
                      /(dx2 *dy2*dz2*(px0p2 + py0p2 + pz0p2) + 2*dx*dy*dz *(dy*dz*px0p \
                      + dx*dz*py0p + dx*dy*pz0p)*T0p + (dy2*dz2 + dx2*(dy2 + dz2))*T0p2)


    # Update traveltime at (y,z,x)=(kp,ip,jp) if ttemp*T0p is smaller than the previously stored value
    if (ttemp*T0p < tau[kp, ip, jp]*T0p):
        tau[kp,ip,jp] = ttemp
  
    return


# Function to run the fast sweeping iterations
def fastsweep_run3d(tau, v, T0, py0, pz0, px0, niter, ny, nz, nx, dy, dz, dx, isy, isz, isx):
    '''
    Runs fast sweeping in 3D media.
    Calls fastsweep_stencil3d() to run the inner loop
    ''' 

    for iloop in range(niter):

        for iy in range(ny):
            for ix in range(nx):
                for iz in range(nz):
                    fastsweep_stencil3d(tau, v, T0, py0, pz0, px0, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in reversed(range(ny)):
            for ix in range(nx):
                for iz in range(nz):
                    fastsweep_stencil3d(tau, v, T0, py0, pz0, px0, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in range(ny):
            for ix in reversed(range(nx)):
                for iz in range(nz):
                    fastsweep_stencil3d(tau, v, T0, py0, pz0, px0, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in reversed(range(ny)):
            for ix in reversed(range(nx)):
                for iz in range(nz):
                    fastsweep_stencil3d(tau, v, T0, py0, pz0, px0, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in range(ny):
            for ix in range(nx):
                for iz in reversed(range(nz)):
                    fastsweep_stencil3d(tau, v, T0, py0, pz0, px0, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in reversed(range(ny)):
            for ix in range(nx):
                for iz in reversed(range(nz)):
                    fastsweep_stencil3d(tau, v, T0, py0, pz0, px0, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in range(ny):
            for ix in reversed(range(nx)):
                for iz in reversed(range(nz)):
                    fastsweep_stencil3d(tau, v, T0, py0, pz0, px0, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in reversed(range(ny)):
            for ix in reversed(range(nx)):
                for iz in reversed(range(nz)):
                    fastsweep_stencil3d(tau, v, T0, py0, pz0, px0, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)
           
    return 