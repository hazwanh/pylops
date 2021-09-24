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
        T[nz, nx]: Numpy array of the traveltime field initialized with infinity everywhere and zero at the source
    ''' 
    
    T = np.ones((nz, nx)) * np.inf
    
    if (0 <= isz <= nz-1) and (0 <= isx <= nx-1):
        T[isz, isx] = 0.
    else:
        raise ValueError(f"sz and sx must be in the ranges z:({oz}, {oz+(nz-1)*dz}) and x:({ox}, {ox+(nx-1)*dx}), respectively")
    
    return T
    
    

# Function to run the inner loop; computes traveltime at each grid point
def fastsweep_stencil2d(T, v, ip, jp, nz, nx, dz, dx, isz, isx):
    '''
    Inputs:
        ip, jp: indices for the considered point along z and x directions, respectively
        nz, nx: total number of grid points along z and x directions, respectively
        dz, dx: grid spacing along z and x directions, respectively
        T[nz, nx]: Numpy array of the traveltime field
    
    Returns: 
        T[nz, nx]: Numpy array of the traveltime field after updating value at (z,x) = (ip,jp)
    '''
    
    # Return without updating for the source point
    if (ip == isz) and (jp == isx):
        return
    
    # Find the minimum valued neighbor along the z direction
    if (ip == 0) or (ip == nz-1):
        if ip == 0:
            tzmin = T[1, jp]
        else:
            tzmin = T[nz-2, jp]
    else:
        if (T[ip-1, jp] < T[ip+1, jp]):
            tzmin = T[ip-1, jp]
        else:
            tzmin = T[ip+1, jp]
    
    # Find the minimum valued neighbor along the x direction
    if (jp ==0) or (jp == nx-1):
        if jp == 0:
            txmin = T[ip, 1]
        else:
            txmin = T[ip, nx-2]
    else:
        if (T[ip, jp-1] < T[ip, jp+1]):
            txmin = T[ip, jp-1]
        else:
            txmin = T[ip, jp+1]
    
    # If both minimum valued neighbors are infinity, return without updating 
    if (txmin == np.inf) and (tzmin == np.inf):
        return
    
    # velocity value at (ip, jp)
    vp = v[ip, jp]
    
    # compute one dimensional traveltime update 
    tz = tzmin + dz/vp
    tx = txmin + dx/vp
        
    if (tz <= txmin): 
        ttemp = tz
    elif (tx <= tzmin):    
        ttemp = tx
    else:
        
        dz2 = dz**2
        dx2 = dx**2
        
        ttemp = (dz*dx*np.sqrt((dz2 + dx2)*(1/vp**2) - (txmin - tzmin)**2) \
                + tzmin*dx2 + txmin*dz2) / (dz2 + dx2);
        

    # Update traveltime at (z,x)=(ip,jp) if ttemp is smaller than the previously stored value
    
    if (ttemp < T[ip, jp]):
        T[ip, jp] = ttemp        
    
    return
    

# Function to run the fast sweeping iterations
def fastsweep_run2d(T, v, niter, nz, nx, dz, dx, isz, isx):
    '''
    Runs fast sweeping in 2D media.
    Calls fastsweep_stencil2d() to run the inner loop
    ''' 

    for iloop in range(niter):
        
        for iz in range(nz):
            for ix in range(nx):
                fastsweep_stencil2d(T, v, iz, ix, nz, nx, dz, dx, isz, isx) 
    
        for iz in range(nz):
            for ix in reversed(range(nx)):
                fastsweep_stencil2d(T, v, iz, ix, nz, nx, dz, dx, isz, isx) 
                
        for iz in reversed(range(nz)):
            for ix in range(nx):
                fastsweep_stencil2d(T, v, iz, ix, nz, nx, dz, dx, isz, isx) 
                
        for iz in reversed(range(nz)):
            for ix in reversed(range(nx)):
                fastsweep_stencil2d(T, v, iz, ix, nz, nx, dz, dx, isz, isx) 
            
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
        T[ny, nz, nx]: Numpy array of the traveltime field initialized with infinity everywhere and zero at the source
    '''
    
    T = np.ones((ny, nz, nx)) * np.inf
    
    if (0 <= isy <= ny-1) and (0 <= isz <= nz-1) and (0 <= isx <= nx-1):
        T[isy, isz, isx] = 0.
    else:
        raise ValueError(f"sy, sz, and sx must be in the ranges \n y:({oy}, {oy+(ny-1)*dy}), \n z:({oz}, {oz+(nz-1)*dz}), and x:({ox}, {ox+(nx-1)*dx}), respectively")
    
    return T
  
  
# Function to run the inner loop; computes traveltime at each grid point
def fastsweep_stencil3d(T, v, kp, ip, jp, ny, nz, nx, dy, dz, dx, isy, isz, isx):
    '''
    Inputs:
        kp, ip, jp: indices for the considered point along y, z, and x directions, respectively
        ny, nz, nx: total number of grid points along y, z, and x directions, respectively
        dy, dz, dx: grid spacing along y, z, and x directions, respectively
        T[ny, nz, nx]: Numpy array of the traveltime field
    
    Returns: 
        T[ny, nz, nx]: Numpy array of the traveltime field after updating value at (y,z,x) = (kp,ip,jp)
    '''
    
    if (kp == isy) and (ip == isz) and (jp == isx):
        return

    # Find the minimim valued neighbor along y direction 

    if (kp == 0) or (kp == ny-1):
        if kp == 0:
            tymin = T[1,ip,jp]; sy=-1
        else:
            tymin = T[ny-2,ip,jp]; sy=1
    else:
        if (T[kp-1,ip,jp] < T[kp+1,ip,jp]):
            tymin = T[kp-1,ip,jp]; sy=1
        else:
            tymin = T[kp+1,ip,jp]; sy=-1      
    
    # Find the minimim valued neighbor along z direction

    if (ip == 0) or (ip == nz-1):
        if ip == 0:
            tzmin = T[kp,1,jp]; sz=-1
        else:
            tzmin = T[kp,nz-2,jp]; sz=1
    else:
        if (T[kp,ip-1,jp] < T[kp,ip+1,jp]):
            tzmin = T[kp,ip-1,jp]; sz=1
        else:
            tzmin = T[kp,ip+1,jp]; sz=-1
    
    # Find the minimim valued neighbor along x direction   

    if (jp == 0) or (jp == nx-1):
        if jp == 0:
            txmin = T[kp,ip,1]; sx=-1
        else:
            txmin = T[kp,ip,nx-2]; sx=1
    else:
        if (T[kp,ip,jp-1] < T[kp,ip,jp+1]):
            txmin = T[kp,ip,jp-1]; sx=1
        else:
            txmin = T[kp,ip,jp+1]; sx=-1
             
    
    if (tymin == np.inf) and (txmin == np.inf) and (tzmin == np.inf):
        return

    # Velocity value at the considered point

    vp = v[kp,ip,jp]

    td = [(txmin,dx),(tymin,dy),(tzmin,dz)]
    td.sort(key=lambda x: x[0])

    tmin = [i[0] for i in td]
    d = [i[1] for i in td]


    ttemp = tmin[0] + d[0]/vp

    if ttemp>tmin[1]:

        ttemp = (d[0]*d[1]*np.sqrt((d[0]**2 + d[1]**2)*(1/vp**2) - (tmin[0] - tmin[1])**2) \
              + tmin[1]*d[0]**2 + tmin[0]*d[1]**2) / (d[0]**2 + d[1]**2);

        if ttemp>tmin[2]:

            dy2 = dy**2
            dz2 = dz**2
            dx2 = dx**2

            ttemp = ((2*txmin)/dx2 + (2*tymin)/dy2 + (2*tzmin)/dz2 + np.sqrt(4*(txmin/dx2 \
                    + tymin/dy2 + tzmin/dz2)**2 - 4*(1/dx2 + 1/dy2 + 1/dz2) \
                    *(txmin**2/dx2 + tymin**2/dy2 + tzmin**2/dz2 - vp**(-2)))) \
                    /(2.*(1/dx2 + 1/dy2 + 1/dz2))


    if (ttemp < T[kp,ip,jp]):
        T[kp,ip,jp] = ttemp
    
    return


# Function to run the fast sweeping iterations
def fastsweep_run3d(T, v, niter, ny, nz, nx, dy, dz, dx, isy, isz, isx):
    '''
    Runs fast sweeping in 3D media.
    Calls fastsweep_stencil3d() to run the inner loop
    ''' 
    
    for iloop in range(niter):

        for iy in range(ny):
            for ix in range(nx):
                for iz in range(nz):
                    fastsweep_stencil3d(T, v, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in reversed(range(ny)):
            for ix in range(nx):
                for iz in range(nz):
                    fastsweep_stencil3d(T, v, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in range(ny):
            for ix in reversed(range(nx)):
                for iz in range(nz):
                    fastsweep_stencil3d(T, v, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in reversed(range(ny)):
            for ix in reversed(range(nx)):
                for iz in range(nz):
                    fastsweep_stencil3d(T, v, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in range(ny):
            for ix in range(nx):
                for iz in reversed(range(nz)):
                    fastsweep_stencil3d(T, v, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in reversed(range(ny)):
            for ix in range(nx):
                for iz in reversed(range(nz)):
                    fastsweep_stencil3d(T, v, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in range(ny):
            for ix in reversed(range(nx)):
                for iz in reversed(range(nz)):
                    fastsweep_stencil3d(T, v, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)

        for iy in reversed(range(ny)):
            for ix in reversed(range(nx)):
                for iz in reversed(range(nz)):
                    fastsweep_stencil3d(T, v, iy, iz, ix, ny, nz, nx, dy, dz, dx, isy, isz, isx)
           
    return 