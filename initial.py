import numpy as np
from numba import jit 

@jit(nopython=True) 
def u_init(u,qu,qc,dn,hs_up,nx,ny,snm,slope,hmin):
    for i in np.arange(0,nx+1):
        qc[i]=0.
        for j in np.arange(1,ny+1):
            hss=hs_up[i,j]
            if hss>hmin:
                u[i,j]=1./snm*hss**(2./3.)*np.sqrt(slope)
            else:
                u[i,j]=0.
            qu[i,j]=u[i,j]*hss*dn[i,j]
            qc[i]=qc[i]+qu[i,j]
    return u        

@jit(nopython=True)     
def h_init(h,hs,hpos_c,eta,nx,ny,hmin):
 for i in np.arange(1,nx+1):
     for j in np.arange(1,ny+1):
        h[i,j]=hpos_c[i]
        hs[i,j]=h[i,j]-eta[i,j]
        if hs[i,j]< hmin:
            hs[i,j]=0.
            h[i,j]=eta[i,j]
 return h,hs

@jit(nopython=True) 
def eta_init(eta,nx,ny,ds,dn,zgrid):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            eta[i,j]=(zgrid[i,j]+zgrid[i-1,j]+zgrid[i,j-1]+zgrid[i-1,j-1])*.25
    return eta

@jit(nopython=True) 
def ep_init(ep,ep_x,nx,ny,snu_0):
    for i in np.arange(0,nx+1):
        for j in np.arange(0,ny+1):
            ep[i,j]=snu_0; ep_x[i,j]=snu_0
    return ep,ep_x

@jit(nopython=True)        
def diffs_init(gux,guy,gvx,gvy,u,v,nx,ny,dsi,dnj,ds,dn):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            gux[i,j]=(u[i+1,j]-u[i-1,j])/(2*dsi[i,j])
            gvx[i,j]=(v[i+1,j]-v[i-1,j])/(2*ds[i,j])
    
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny):
            guy[i,j]=(u[i,j+1]-u[i,j-1])/(2*dn[i,j])
            gvy[i,j]=(v[i,j+1]-v[i,j-1])/(2*dnj[i,j])

    return gux,guy,gvx,gvy




