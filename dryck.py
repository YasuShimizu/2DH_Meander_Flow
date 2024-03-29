import numpy as np
from numba import jit

@jit
def u_ck(un,qu,hs,hmin,gux,guy,nx,ny):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            if hs[i,j]<hmin and hs[i+1,j]<hmin:
                un[i,j]=0.; gux[i,j]=0.; guy[i,j]=0.;qu[i,j]=0.
    return un,qu,gux,guy

@jit
def v_ck(vn,qv,hs,hmin,gvx,gvy,nx,ny):
    for j in np.arange(1,ny):
        for i in np.arange(1,nx+1):
            if hs[i,j]<hmin and hs[i,j+1]<hmin:
                vn[i,j]=0.; gvx[i,j]=0.; gvy[i,j]=0.;qv[i,j]=0.
    return vn,qv,gvx,gvy
