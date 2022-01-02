import numpy as np
from numba import jit

def center(nx,t0_degree,s0,x0,y0,lam,dds,xpos,ypos,thepos,spos):
    theta0=np.radians(t0_degree)
    s=s0; x=x0; y=y0
    xpos[0]=x; ypos[0]=y
    thepos[0]=theta0*np.sin(2.*np.pi*s/lam)
    for i in np.arange(1,nx+1):
        for j in np.arange(1,11):
            s=s+dds
            theta=theta0*np.sin(2.*np.pi*s/lam)
            x=x+dds*np.cos(theta)
            y=y+dds*np.sin(theta)
        xpos[i]=x; ypos[i]=y; thepos[i]=theta; spos[i]=s
    return xpos,ypos,thepos,spos

def sggrid(nx,chb,chl,slope,lam,amp,delta,xpos,ypos,spos,thepos,ny,dz,xr,yr,xl,yl,xgrid,ygrid,zgrid):
    eta0=chl*slope
    for i in np.arange(0,nx+1):
        zcenter=eta0-spos[i]*slope
        xr[i]=xpos[i]+chb/2.*np.sin(thepos[i])
        yr[i]=ypos[i]-chb/2.*np.cos(thepos[i])
        xl[i]=xpos[i]-chb/2.*np.sin(thepos[i])
        yl[i]=ypos[i]+chb/2.*np.cos(thepos[i])
    
        for j in np.arange(0,ny+1):
            ss=float(j)/float(ny)
            xgrid[i,j]=xr[i]+ss*(xl[i]-xr[i])
            ygrid[i,j]=yr[i]+ss*(yl[i]-yr[i])
            dz[i,j]=-amp*np.cos(2.*np.pi/lam*(spos[i]-delta))*np.cos(np.pi*ss)
            zgrid[i,j]=dz[i,j]+zcenter
    return xr,yr,xl,yl,xgrid,ygrid,zgrid

