from types import SimpleNamespace
import numpy as np
from numba import jit

@jit
def down(nx,ny,dn,eta,qp,snm,hs0,slope):
    h_max=np.max(eta)+hs0; h_min=np.min(eta)
    eps=qp;epsmin=qp/100.
#    print(eps,epsmin)
    while eps>epsmin:
        h_down=(h_max+h_min)*.5
        qcd=0.
        for j in np.arange(1,ny+1):
            hs1=h_down-eta[nx,j]
            if hs1<0.:
                hs1=0.
                u01=0.
            else:
                u01=1./snm*hs1**(2./3.)*np.sqrt(slope)
                qcd=qcd+u01*hs1*dn[nx,j]
        eps=np.abs(qp-qcd)
        if qcd>qp:
            h_max=h_down
        else:
            h_min=h_down
    return h_down    

@jit
def h_line(hpos_c,eta,spos_c,h_down,slope,nx,nym,hmin):
    tlen=spos_c[nx]
    for i in np.arange(1,nx+1):
        hpos_c[i]=h_down+(tlen-spos_c[i])*slope
        hs_c=hpos_c[i]-eta[i,nym]
        if hs_c< hmin:
            hpos_c[i]=eta[i,nym]
    return hpos_c


@jit
def h_uniform(hpos_c,nx,ny,dn,eta,qp,snm,hs0,h_down,hmin,slope):
    hpos_c[nx]=h_down
    epsmin=qp/1000.
    for i in np.arange(1,nx):
        h_max=np.max(eta[i,:])+hs0; h_min=np.min(eta[i,:])
        eps=qp
        while eps>epsmin:
            hpos_c[i]=(h_max+h_min)*.5
            qcd=0.
            for j in np.arange(1,ny+1):
                hs1=hpos_c[i]-eta[i,j]
                if hs1<hmin:
                    hs1=0.
                    u01=0.
                else:
                    u01=1./snm*hs1**(2./3.)*np.sqrt(slope)
                    dni=(dn[i,j]+dn[i-1,j])*.5
                    qcd=qcd+u01*hs1*dni
            eps=np.abs(qp-qcd)
            if qcd>qp:
                h_max=hpos_c[i]
            else:
                h_min=hpos_c[i]
    return hpos_c


@jit
def h_nonuni(hpos_c,c_area,e_slope,alf_f,eta,qp,spos_c,hs0,h_down,nx,ny,nym,ds,dn,snm,hmin,g):
    hpos_c[nx]=h_down
    epsmin=hmin/100.
    for i in np.arange(nx,1,-1):
        c_area[i]=0.;b1=0.; b2=0.
        for j in np.arange(1,ny+1):
            hs1=hpos_c[i]-eta[i,j]
            if hs1>0.:
                dnn=(dn[i,j]+dn[i-1,j])*.5
                c_area[i]=c_area[i]+hs1*dnn
                b1=b1+dnn*hs1**3/snm**3
                b2=b2+dnn*hs1**(5./3.)/snm
        alf_f[i]=b1/b2**3
        e_slope[i]=qp**2/b2**2
        if i>1:
            hpos_c[i-1]=hpos_c[i]
            eps=hs0; nc=0
            while eps>epsmin and nc<500:
                c_area[i-1]=0.
                b1=0.;b2=0.
                for j in np.arange(1,ny+1):
                    hs1=hpos_c[i-1]-eta[i-1,j]
                    if hs1>0.:
                        dnn=(dn[i-1,j]+dn[i-2,j])*.5
                        c_area[i-1]=c_area[i-1]+hs1*dnn
                        b1=b1+dnn*hs1**3/snm**3
                        b2=b2+dnn*hs1**(5./3.)/snm
                alf_f[i-1]=b1/b2**3
                e_slope[i-1]=qp**2/b2**2
                dsx=(ds[i,nym]+ds[i-1,nym])*.5
                h_a1=hpos_c[i]+qp**2/(2.*g)*(alf_f[i]-alf_f[i-1])+dsx*.5*(e_slope[i]+e_slope[i-1])
                eps=np.abs(h_a1-hpos_c[i-1])
                nc=nc+1
                hpos_c[i-1]=h_a1
    return hpos_c
