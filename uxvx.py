import numpy as np

def uv(ux,vx,uv2,hx,hsx,u,v,h,hs,nx,ny,coss,sins):
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny):
            ux[i,j]=(u[i,j]+u[i,j+1])*.5
        ux[i,0]=u[i,1]; ux[i,ny]=u[i,ny]
    
    
    for j in np.arange(0,ny+1):
        for i in np.arange(1,nx):    
            vx[i,j]=(v[i,j]+v[i+1,j])*.5
        vx[0,j]=v[1,j]; vx[nx,j]=v[nx,j]

    for i in np.arange(0,nx+1):
        for j in np.arange(0,ny+1):
            utmp=ux[i,j]*coss[i,j]-vx[i,j]*sins[i,j]
            vtmp=ux[i,j]*sins[i,j]+vx[i,j]*coss[i,j]
            ux[i,j]=utmp; vx[i,j]=vtmp

    uv2[:,:]=np.sqrt(ux[:,:]**2+vx[:,:]**2)

    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            hx[i,j]=(h[i,j]+h[i+1,j]+h[i,j+1]+h[i+1,j+1])*.25
            hsx[i,j]=(hs[i,j]+hs[i+1,j]+hs[i,j+1]+hs[i+1,j+1])*.25
        hx[i,0]=(h[i,1]+h[i+1,1])*.5
        hx[i,ny]=(h[i,ny]+h[i+1,ny])*.5
        hsx[i,0]=(hs[i,1]+hs[i+1,1])*.5
        hsx[i,ny]=(hs[i,ny]+hs[i+1,ny])*.5
    for j in np.arange(1,ny):
        hx[0,j]=(h[1,j]+h[1,j+1])*.5
        hx[nx,j]=(h[nx,j]+h[nx,j+1])*.5
        hsx[0,j]=(hs[1,j]+hs[1,j+1])*.5
        hsx[nx,j]=(hs[nx,j]+hs[nx,j+1])*.5
    hx[0,0]=h[1,1];hx[0,ny]=h[1,ny];hx[nx,0]=h[nx,1];hx[nx,ny]=h[nx,ny]
    hsx[0,0]=hs[1,1];hsx[0,ny]=hs[1,ny];hsx[nx,0]=hs[nx,1];hsx[nx,ny]=hs[nx,ny]
    return ux,vx,uv2,hx,hsx

def vortex(vor,ux,vx,nx,ny,ds,dn):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            vor[i,j]=(ux[i,j+1]-ux[i,j-1])/(dn[i,j+1]+dn[i,j])- \
                     (vx[i+1,j]-vx[i-1,j])/(ds[i,j]+ds[i+1,j])

    return vor

