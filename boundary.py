import numpy as np
from numba import jit

@jit
def h_bound(h,hs,eta,nx,ny,j_west,j_east,j_hdown,h_down,hmin):
    if j_west==1: #Upstream Boundary
        h[0]=h[1];hs[0]=hs[1] #Upstream Wall
    if j_east==1: #Downstream Boundary
        h[nx]=h[nx-1]; hs[nx]=h[nx]-eta[nx] #Downstream Wall
    else: #Downstream Free
        if j_hdown>=1: #Stage Given
            for j in np.arange(1,ny+1):
                h[nx,j]=h_down; hs[nx,j]=h[nx,j]-eta[nx,j]
                if hs[nx,j]<hmin:
                    hs[nx,j]=0.
                    h[nx,j]=eta[nx,j]
        else: #Free Downstream
            hs[nx]=hs[nx-1];h[nx]=hs[nx]+eta[nx] 
    return h,hs

@jit
def u_bound(u,nx,ny,j_west,j_east,ijh,uinp,hs,hmin):
    if j_west==1: #Upstream Velocity boundary
        u[0]=0. #Upstream Wall
    else:
        for j in np.arange(1,ny+1):
            u[0,j]=uinp[j]
    if j_east==1: #Downstream 
        u[nx]=0.
    else:
        u[nx]=u[nx-1]
    for j in np.arange(1,ny+1):
#        for i in np.arange(nx-3,nx+1):
#            if u[i,j]<0:
#                u[i,j]=0.
        if hs[nx,j]<hmin:
            u[nx,j]=0.
    return u

@jit
def v_bound(v,nx,ny,ijh):
    for i in np.arange(1,nx+1):
        v[i,0]=0.; v[i,ny]=0.
        for j in np.arange(1,ny):
            if ijh[i,j]>0 or ijh[i,j+1]>0:
                v[i,j]=0.
    return v

@jit
def u_upstream(uinp,u,hs,qu,qc,snm,slope,dn,qp,ny,hmin):
    qc[0]=0.
    for j in np.arange(1,ny+1):
        if hs[1,j]<hmin:
            uinp[j]=0.;qu[0,j]=0.
        else:
            uinp[j]=1./snm*hs[1,j]**(2./3.)*np.sqrt(slope)
            qu[0,j]=uinp[j]*hs[1,j]*dn[0,j]
            qc[0]=qc[0]+qu[0,j]
    qdiff=qc[0]/qp
    qc[0]=0.
    for j in np.arange(1,ny+1):
        uinp[j]=uinp[j]/qdiff
        u[0,j]=uinp[j]
        qu[0,j]=qu[0,j]/qdiff
        qc[0]=qc[0]+qu[0,j]
    return uinp,u,qu,qc

@jit
def gbound_u(gux,guy,ijh,nx,ny):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            if ijh[i,j]>0 or ijh[i+1,j]>0:
                gux[i,j]=0.; guy[i,j]=0.
    return gux,guy

@jit
def gbound_v(gvx,gvy,ijh,nx,ny):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            if ijh[i,j]>0 or ijh[i,j+1]>0:
                gvx[i,j]=0.; gvy[i,j]=0.
    return gvx,gvy
