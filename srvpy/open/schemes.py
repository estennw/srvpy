"""
This submodule contains base tools for numerical integration of the Hamilton-
Jacobi-Bellman equation in the Square Root Velocity Framework. The methods
accepts and returns vectorized data only.
"""

# External dependencies:
import numpy as np



class Scheme:
    """
    Object containing at least two methods: 'update' and 'alpha'.

    Methods
    -------
    update(u,reg,i,j):
        Computes the value function u[i,j] by approximating local solutions
        to the HJB equation.
    alpha(u,reg,i,j):
        Computes the maximizer alpha corresponding to the solution of the
        approximated HJB equation.
    """
    filterCompatible = False
    eps = np.finfo(float).eps


class Filter(Scheme):

    def __init__(self,ascheme,mscheme,tol=1,order=.5):
        assert ascheme.type == mscheme.type, "Solvers must be of same type"
        self.type = ascheme.type
        self.ascheme = ascheme
        self.mscheme = mscheme
        self.tol = tol
        self.order = order

    def update(self,u,hjbdata,i,j):
        self.ascheme.update(u,hjbdata,i,j)
        H = self.mscheme.hamiltonian(u,hjbdata,i,j)
        H[u[i,j]<np.maximum(u[i,j-1],u[i-1,j])] = np.inf
        x1,x2 = hjbdata.x1,hjbdata.x2
        b = np.abs(H) > self.tol*np.sqrt((x1[i]-x1[i-1])*(x2[j]-x2[j-1]))**(1+self.order)
        self.mscheme.update(u,hjbdata,i[b],j[b])

    def alpha(self,u,hjbdata,i,j):
        H = self.mscheme.hamiltonian(u,hjbdata,i,j)
        x1,x2 = hjbdata.x1,hjbdata.x2
        if np.abs(H) > self.tol*np.sqrt((x1[i]-x1[i-1])*(x2[j]-x2[j-1]))**(1+self.order):
            return self.mscheme.alpha(u,hjbdata,i,j)
        return self.ascheme.alpha(u,hjbdata,i,j)


class Maximal(Scheme):

    def __init__(self,*schemes):
        self.type = schemes[0].type
        #assert ascheme.type == mscheme.type, "Solvers must be of same type"
        self.schemes = schemes

    def update(self,u,hjbdata,i,j):
        for scheme in self.schemes:
            temp = u[i,j]
            scheme.update(u,hjbdata,i,j)
            u[i,j] = np.maximum(temp,u[i,j])

    def alpha(self,u,hjbdata,i,j):
        return self.schemes[0].alpha(u,hjbdata,i,j)


class DDP(Scheme):
    type = "u"
    monotone = True
    def update(u,hjbdata,i,j):
        i2 = np.maximum(i[:,None]-hjbdata.nbhd[[0]],0)
        j2 = np.maximum(j[:,None]-hjbdata.nbhd[[1]],0)
        u[i,j] = (u[i2,j2] + hjbdata[i[:,None],j[:,None],i2,j2]).max(axis=1)
    def prev(u,hjbdata,i,j):
        i,j = np.array([i]),np.array([j])
        i2 = np.maximum(i-hjbdata.nbhd[[0]],0)
        j2 = np.maximum(j-hjbdata.nbhd[[1]],0)
        optimal = (u[i2,j2] + hjbdata[i,j,i2,j2]).argmax(axis=1)
        return [i2[np.arange(i.shape[0]),optimal], j2[np.arange(j.shape[0]),optimal]]


class U1(Scheme):
    type = "u"
    monotone = True
    def update(u,hjbdata,i,j):
        u01,u10 = u[i-1,j],u[i,j-1]
        u[i,j] = 0.5*(u01 + u10 + np.sqrt((u01 - u10)**2 + hjbdata[i,j]**2))
    def hamiltonian(u,hjbdata,i,j):
        u01,u10 = u[i-1,j],u[i,j-1]
        return 0.5*(u01 + u10 + np.sqrt((u01 - u10)**2 + hjbdata[i,j]**2)) - u[i,j]
    def alpha(u,hjbdata,i,j):
        u01,u10 = u[i-1,j],u[i,j-1]
        fh2 = hjbdata[i,j]**2
        return (1 + (u01-u10)/np.sqrt((u01-u10)**2 + fh2 + Scheme.eps),
                1 + (u10-u01)/np.sqrt((u01-u10)**2 + fh2 + Scheme.eps))

class UInf(Scheme):
    type = "u"
    monotone = True
    def update(u,hjbdata,i,j):
        u0,u1 = u[i-1,j-1],np.maximum(u[i-1,j],u[i,j-1])
        fh = hjbdata[i,j]
        ufh2 = np.minimum(0.25*fh**2,(u1-u0)**2)
        u[i,j] = np.maximum(u1 + ufh2/(u1-u0+Scheme.eps),u0+fh)
    def hamiltonian(u,hjbdata,i,j):
        u0,u1 = u[i-1,j-1],np.maximum(u[i-1,j],u[i,j-1])
        fh = hjbdata[i,j]
        ufh2 = np.minimum(0.25*fh**2,(u1-u0)**2)
        return np.maximum(u1 + ufh2/(u1-u0+Scheme.eps),u0+fh**2) - u[i,j]
    def alpha(u,hjbdata,i,j):
        u00,u01,u10 = u[i-1,j-1],u[i-1,j],u[i,j-1]
        fh2 = hjbdata[i,j]**2
        if u01>=u10:
            return 1,min(1,fh2/(4*(u01-u00)**2+Scheme.eps))
        return min(1,fh2/(4*(u10-u00)**2+Scheme.eps)),1

class U2(Scheme):
    type = "u"
    monotone = False
    def update(u,hjbdata,i,j):
        u[i,j] = u[i-1,j-1] + np.sqrt((u[i-1,j] - u[i,j-1])**2 + hjbdata[i,j]**2)
    def alpha(u,hjbdata,i,j):
        u01,u10 = u[i-1,j],u[i,j-1]
        fh2 = hjbdata[i,j]**2
        return (1 + (u01-u10)/np.sqrt((u01-u10)**2 + fh2 + Scheme.eps),
                1 + (u10-u01)/np.sqrt((u01-u10)**2 + fh2 + Scheme.eps))



class V1(Scheme):
    type = "v"
    monotone = True
    def update(v,hjbdata,i,j):
        v01,v10 = v[i-1,j],v[i,j-1]
        fh2 = hjbdata[i,j]**2
        v[i,j] = 0.5*(v01 + v10 + fh2 +
                      np.sqrt((v01 - v10)**2 +
                              2*(v01 + v10)*fh2 +
                              fh2**2))
    def hamiltonian(v,hjbdata,i,j):
        v01,v10,v11 = v[i-1,j],v[i,j-1],v[i,j]
        v01 /= 2*np.sqrt(v11)+Scheme.eps
        v10 /= 2*np.sqrt(v11)+Scheme.eps
        v11 /= 2*np.sqrt(v11)+Scheme.eps
        return 0.5*(v01+v10 + np.sqrt((v01 - v10)**2 + hjbdata[i,j]**2)) - v11
    def alpha(v,hjbdata,i,j):
        v01 = v[i-1,j]
        v10 = v[i,j-1]
        v11 = v[i,j]
        fh2 = hjbdata[i,j]**2
        return (1 + (v01-v10)/np.sqrt((v01-v10)**2 + 4*v11*fh2 + Scheme.eps),
                1 + (v10-v01)/np.sqrt((v01-v10)**2 + 4*v11*fh2 + Scheme.eps))


class VInf(Scheme):
    type = "v"
    monotone = True
    def update(v,hjbdata,i,j):
        fh = hjbdata[i,j]
        fh2 = fh**2
        v0 = v[i-1,j-1]
        v1 = np.maximum(v[i-1,j],v[i,j-1])
        stable = v1*fh2 < (v1-v0)*(v1-v0-fh2)
        v11 = (fh + np.sqrt(fh2+v0))**2
        v11[stable] = (v1[stable]*(v1[stable]-v0[stable]))/(v1[stable]-v0[stable]-fh2[stable]+Scheme.eps)
        v[i,j] = np.maximum(v11,v1)
    def hamiltonian(v,hjbdata,i,j):
        v0,v1,v11 = v[i-1,j-1],np.maximum(v[i,j-1],v[i-1,j]),v[i,j]
        v0 /= 2*np.sqrt(v11)+Scheme.eps
        v1 /= 2*np.sqrt(v11)+Scheme.eps
        v11 /= 2*np.sqrt(v11)+Scheme.eps
        fh = hjbdata[i,j]
        ufh2 = np.minimum(0.25*fh**2,(v1-v0)**2)
        return np.maximum(v1 + ufh2/(v1-v0+Scheme.eps),v0+fh) - v11
    def alpha(v,hjbdata,i,j):
        v00,v01,v10,v11 = v[i-1,j-1],v[i-1,j],v[i,j-1],v[i,j]
        v00 /= 2*np.sqrt(v11)+Scheme.eps
        v01 /= 2*np.sqrt(v11)+Scheme.eps
        v10 /= 2*np.sqrt(v11)+Scheme.eps
        fh2 = hjbdata[i,j]**2
        if v01>=v10:
            a1,a2 = 1,min(1,fh2/(4*(v01-v00)**2+Scheme.eps))
        else:
            a1,a2 = min(1,fh2/(4*(v10-v00)**2+Scheme.eps)),1
        return a1/(a1+a2), a2/(a1+a2)


class V2b(Scheme):
    type = "v"
    monotone = False
    def update(v,hjbdata,i,j):
        v00,v01,v10 = v[i-1,j-1],v[i-1,j],v[i,j-1]
        fh2 = hjbdata[i,j]**2
        v[i,j] = np.maximum.reduce([
            v00 + np.sqrt((v01 - v10)**2 + 2*(v01+v10)*hjbdata[i,j]**2),
            v01,
            v10
        ])
    def alpha(v,hjbdata,i,j):
        v01,v10 = v[i-1,j],v[i,j-1]
        fh2 = 2*(v01+v10)*hjbdata[i,j]**2
        return (1 + (v01-v10)/np.sqrt((v01-v10)**2 + fh2 + Scheme.eps),
                1 + (v10-v01)/np.sqrt((v01-v10)**2 + fh2 + Scheme.eps))


class V2(Scheme):
    type = "v"
    monotone = False
    def update(v,hjbdata,i,j):
        v00,v01,v10 = v[i-1,j-1],v[i-1,j],v[i,j-1]
        fh2 = hjbdata[i,j]**2
        v[i,j] = np.maximum.reduce([
            v00+0.5*fh2+np.sqrt((v01-v10)**2+fh2*(2*v00+v10+v01+0.25*fh2)),
            v01,
            v10
        ])
    def alpha(v,hjbdata,i,j):
        v01,v10 = v[i-1,j],v[i,j-1]
        fh2 = 2*(v01+v10)*hjbdata[i,j]**2
        return (1 + (v01-v10)/np.sqrt((v01-v10)**2 + fh2 + Scheme.eps),
                1 + (v10-v01)/np.sqrt((v01-v10)**2 + fh2 + Scheme.eps))





class Vb24(Scheme):
    type = "v"
    monotone = False
    def update(v,hjbdata,i,j):
        v00,v01,v10 = v[i-1,j-1],v[i-1,j],v[i,j-1]
        fh2 = hjbdata[i,j]**2
        v[i,j] = v00+0.5*fh2+np.sqrt((v01-v10)**2+fh2*(2*v00+v10+v01+0.25*fh2))
    def alpha(v,hjbdata,i,j):
        v01,v10 = v[i-1,j],v[i,j-1]
        fh2 = 2*(v01+v10)*hjbdata[i,j]**2
        return (1 + (v01-v10)/np.sqrt((v01-v10)**2 + fh2 + Scheme.eps),
                1 + (v10-v01)/np.sqrt((v01-v10)**2 + fh2 + Scheme.eps))

class Vb22(Scheme):
    type = "v"
    monotone = False
    def update(v,hjbdata,i,j):
        v00,v01,v10 = v[i-1,j-1],v[i-1,j],v[i,j-1]
        fh2 = hjbdata[i,j]**2
        v[i,j] = v00+fh2+np.sqrt((v01-v10)**2+fh2*(4*v00+fh2))
    def alpha(v,hjbdata,i,j):
        v00,v01,v10,v11 = v[i-1,j-1],v[i-1,j],v[i,j-1],v[i,j]
        fh2 = 2*(v00+v11)*hjbdata[i,j]**2
        return (1 + (v01-v10)/np.sqrt((v01-v10)**2 + fh2 + Scheme.eps),
                1 + (v10-v01)/np.sqrt((v01-v10)**2 + fh2 + Scheme.eps))

class Vb23(Scheme):
    type = "v"
    monotone = False
    def update(v,hjbdata,i,j):
        v[i,j] = (v[i-1,j-1]**.5 + np.sqrt((v[i-1,j]**.5 - v[i,j-1]**.5)**2 + hjbdata[i,j]**2))**2
    def alpha(v,hjbdata,i,j):
        u01,u10 = v[i-1,j]**.5,v[i,j-1]**.5
        fh2 = hjbdata[i,j]**2
        return (1 + (u01-u10)/np.sqrt((u01-u10)**2 + fh2 + Scheme.eps),
                1 + (u10-u01)/np.sqrt((u01-u10)**2 + fh2 + Scheme.eps))
