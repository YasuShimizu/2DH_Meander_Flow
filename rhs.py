import numpy as np
from numba import jit

@jit
def un_cal(un,u,nx,ny,dsi,cfx,ctrx,hn,g,dt,hmin,hs,eta):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            if hs[i,j]<hmin or hs[i+1,j]<hmin:
                dhdx=0.
            else:
                dhdx=(hn[i+1,j]-hn[i,j])/dsi[i,j]
            un[i,j]=u[i,j]+(ctrx[i,j]+cfx[i,j]-g*dhdx)*dt
    return un

@jit
def vn_cal(vn,v,nx,ny,dnj,cfy,ctry,hn,g,dt,hmin,hs,eta):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            if hs[i,j]<hmin or hs[i,j+1]<hmin:
                dhdy=0.
            else:
                dhdy=(hn[i,j+1]-hn[i,j])/dnj[i,j]
            vn[i,j]=v[i,j]+(ctry[i,j]+cfy[i,j]-g*dhdy)*dt
    return vn

@jit
def qu_cal(qu,qc,un,nx,ny,dn,hs_up,hmin):
    for i in np.arange(1,nx+1):
        qc[i]=0.
        for j in np.arange(1,ny+1):
            if hs_up[i,j]<hmin:
                qu[i,j]=0.
            else:
                qu[i,j]=un[i,j]*dn[i,j]*hs_up[i,j]
                qc[i]=qc[i]+qu[i,j]
    return qu,qc

@jit
def qv_cal(qv,vn,nx,ny,ds,hs_vp,hmin):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            if hs_vp[i,j]<hmin:
               qv[i,j]=0.
            else:
                qv[i,j]=vn[i,j]*ds[i,j]*hs_vp[i,j]
    return qv

@jit
def qc_cal(qu,qc,nx,ny):
    for i in np.arange(1,nx+1):
        qc[i]=0.
        for j in np.arange(1,ny+1):
            qc[i]=qc[i]+qu[i,j]
    return qc
    