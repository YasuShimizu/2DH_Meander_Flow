from numba.types.npytypes import NumpyNdIterType
import numpy as np
from numba import jit

@jit(nopython=True)
def dsdncal(nx,ny,ds,dn,dsi,dnj,rho_r,rho_s,rho_n,coss,sins,area,xgrid,ygrid):
    for i in np.arange(0,nx+1):
        for j in np.arange(0,ny+1):
            ds[i,j]=np.sqrt((xgrid[i,j]-xgrid[i-1,j])**2+(ygrid[i,j]-ygrid[i-1,j])**2)
            coss[i,j]=(xgrid[i,j]-xgrid[i-1,j])/ds[i,j]
                
    for j in np.arange(0,ny+1):
        for i in np.arange(0,nx+1):    
            dn[i,j]=np.sqrt((xgrid[i,j]-xgrid[i,j-1])**2+(ygrid[i,j]-ygrid[i,j-1])**2)
            sins[i,j]=(ygrid[i,j]-ygrid[i-1,j])/ds[i,j]

    for j in np.arange(0,ny+1):
        for i in np.arange(1,nx):
            rho_r[i,j]=(coss[i,j]*(sins[i+1,j]-sins[i,j])-sins[i,j]*(coss[i+1,j]-coss[i,j]))/ds[i,j]


    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            dsi[i,j]=(ds[i,j]+ds[i,j-1]+ds[i+1,j]+ds[i+1,j-1])*.25
    
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            dnj[i,j]=(dn[i,j]+dn[i-1,j]+dn[i,j+1]+dn[i-1,j+1])*.25

    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            area[i,j]=(ds[i,j]+ds[i,j-1])*(dn[i,j]+dn[i-1,j])*.25

    return ds,dn,dsi,dnj,rho_r,rho_s,rho_n,coss,sins,area
    
@jit(nopython=True)
def center(nx,nym,dsi,spos_c):
    spos_c[1]=0.
    for i in np.arange(2,nx+1):
        spos_c[i]=spos_c[i-1]+dsi[i-1,nym]
    return spos_c

