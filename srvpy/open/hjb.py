"""
This submodule contains objects for the numerical registration of curves and
computation of shape space geodesics in the Square Root Velocity Framework.
"""

# External dependencies:
import numpy as np
from math import gcd, ceil, floor
from scipy.interpolate import interp1d
import numbers

# Internal dependencies:
from .schemes import *





class HJBSolver:
    """
    Class with necessary methods to register curves and compute geodesics in
    the Square Root Velocity Framework.
    """

    def __init__(self, scheme):
        """
        Initializes a solver for the HJB equation.

        Parameters
        scheme : Scheme
            The scheme used by the solver
        """

        self.scheme = scheme


    def solve_hjb(self):
        """
        Approximates viscosity solutions of the Hamilton-Jacobi-Bellman for the
        value function.

        Returns
        -------
        value : ndarray (n+1,m+1)
            Approximations of the value function on the grid.
        """

        # Initialize the value function
        self.value = np.zeros(self.shape)
        n = self.shape[0]-1
        m = self.shape[1]-1

        # Iterate through the anti-diagonals of the array.
        for k in range(2,n+m+1):
            i = np.arange(max(k-m,1), min(k,n+1))
            j = np.arange(max(k-n,1), min(k,m+1))[::-1]
            self.value[i,j] = self.scheme.solve(self,i,j)

        return self.value


    def solve_hjb_strip(self,w=None):
        """
        Approximates viscosity solutions of the Hamilton-Jacobi-Bellman for the
        value function.
        """

        # Initialize the value function
        self.value = np.zeros(self.shape)
        n = self.shape[0]-1
        m = self.shape[1]-1

        assert n==m, "The strip-based solver only works with "

        if w is None:
            w = n

        # Iterate through the anti-diagonals of the array.
        for k in range(2,2*n+2):
            wk = np.min([w,k-2,2*n+2-k])
            i = np.arange(int(floor(max((k-wk)/2,1))), int(ceil(min((k+wk)/2,n)+1)))
            j = i[::-1]
            self.scheme.update(self,i,j)

        return self.value


    def _backtrack_hjb_pwc(self,phi1=None):
        """
        Backtracks the HJB equation.

        Parameters
        ----------
        phi1 : arraylike (2,), optional
            The terminal condition (and starting point) of the backtracking. If
            not specified, phi1 is set to (1,1).

        Returns
        -------
        phi1, phi2 : ndarray (2*n+1,)
            Approximations of the optimal reparametrisation path
        """

        n = self.shape[0]-1
        m = self.shape[1]-1
        if phi1 is not None:
            g = np.zeros((phi1[0]+phi1[1]+1,2))
            phi[-1] = phi1
        else:
            phi = np.zeros((n+m+1,2))
            phi[-1] = (1,1)
        eps = np.finfo(float).eps

        i = n-1
        j = m-1
        for k in reversed(range(1,n+m+1)):

            # Check if we have reached origo
            if (phi[k]==[0,0]).all():
                phi = phi[k:]
                break

            # Chech if we have reached the south or west boundary
            if phi[k,0]==0:
                phi[k-1] = 0,self.x2[j]
                j = j-1
                continue
            if phi[k,1]==0:
                phi[k-1] = self.x1[i],0
                i = i-1
                continue

            h1 = phi[k,0]-self.x1[i]
            h2 = phi[k,1]-self.x2[j]

            a1,a2 = self.scheme.alpha(self.value,self,i+1,j+1)
            if a1==a2==0:
                a1 = 1

            b = h2*a1 - h1*a2
            if a2<=0 or phi[k,1]==0:
                phi[k-1] = self.x1[i],phi[k,1]
            elif a1<=0 or phi[k,0]==0:
                phi[k-1] = phi[k,0],self.x2[j]
            elif b>=0:
                phi[k-1] = self.x1[i], self.x2[j] + b/a1
            else:
                phi[k-1] = self.x1[i] - b/a2, self.x2[j]

            if phi[k-1][0] == self.x1[i]:
                i = i-1
            if phi[k-1][1] == self.x2[j]:
                j = j-1

        return phi


    def _backtrack_hjb_constL1(self,phi1=None):
        """
        Backtracks the HJB equation.

        Parameters
        ----------
        phi1 : arraylike (2,), optional
            The terminal condition (and starting point) of the backtracking. If
            not specified, phi1 is set to (1,1).
        """

        n = self.shape[0]-1
        m = self.shape[1]-1
        psi = np.zeros(n+m+1)
        phi = np.zeros((n+m+1,2))
        phi[-1,:] = (n,m)
        eps = np.finfo(float).eps

        for k in reversed(range(1,n+m+1)):

            # Chech if we have reached the south or west boundary
            if k+psi[k]==0:
                psi[k-1] = -(k-1)
                continue
            if k-psi[k]==0:
                psi[k-1] = k-1
                continue

            phi1 = 0.5*(k+psi[k])
            phi2 = 0.5*(k-psi[k])
            i = int(floor(phi1))
            j = int(floor(phi2))
            if phi1 == i:
                dpsi = np.diff(self.scheme.alpha(self.value,self,i,j))
            else:
                dpsir = np.diff(self.scheme.alpha(self.value,self,i+1,j))
                dpsil = np.diff(self.scheme.alpha(self.value,self,i,j+1))
                h1 = phi1-i
                dpsi = h1*dpsir + (1-h1)*dpsil

            psi[k-1] = np.clip(psi[k]+dpsi,-(k-1),k-1)

        k = np.arange(n+m+1)
        phi = np.c_[interp1d(2*np.arange(n+1),self.x1)(k+psi),
                    interp1d(2*np.arange(m+1),self.x2)(k-psi)]

        return phi
    backtrack_hjb = _backtrack_hjb_constL1

    def register(self):
        self.solve_hjb()
        phi = self.backtrack_hjb()
        self.resample(phi[:,0],phi[:,1])

    def f(self, i, j):
        return np.maximum(self.ip(self.q1sdt[i-1],self.q2sdt[j-1]),0)

    def __getitem__(self, i):
        """
        Returns max(fij * sqrt(dxi * dyj), 0).
        """
        return np.maximum(self.ip(self.q1sdt[i[0]-1],self.q2sdt[i[1]-1]),0)


    def objective(self):
        """
        Computes the objective value of the problem along the identity path.
        """
        return self.ip(self.q1sdt,self.q2sdt).sum()




class HJBSolverQ(HJBSolver):
    """
    Class for solving the HJB equation when point evaluations of the SRVTs of
    the curves are available.
    """


    def __init__(self,q1,q2,x1,x2=None,scheme=VInf,unit_sphere=False,normalize=False):
        """
        Initializes a solver for the HJB equation based on SRVTs of the curves.

        Parameters
        scheme : Scheme
            The scheme used by the solver
        q1, q2 : callable
            Functions representing the SRVTs of the curves. The functions
            accepts an ndarray (n,) and return either a real or complex
            ndarray (n,) or a real or complex ndarray (n,d).
        x1, x2 : ndarray (n,) or int
            Grids used for solving the HJB equation. If x1 is int, then the
            a uniform grid with x1+1 points will be used. Similar for x2.
        unit_sphere : bool
            If True, we use the unit sphere topology to compute geodesics.
            If False, we use the L2 topology.
        normalize : bool
            If True, the curves are normalized to be approximately unit length.
        """

        # Construct the grids
        if isinstance(x1, numbers.Integral):
            x1 = np.linspace(0,1,x1+1)
        if x2 is None:
            x2 = x1
        elif isinstance(x2, numbers.Integral):
            x2 = np.linspace(0,1,x2+1)

        self.q1 = q1
        self.q2 = q2
        self.scheme = scheme
        self.unit_sphere = unit_sphere
        self.normalize = normalize

        # Find the dimension of the curves
        q101 = q1(np.array([0,1]))
        q201 = q2(np.array([0,1]))

        # Specify the according inner product
        if q101.ndim == 1 and q201.ndim == 1:
            self.ndim = 1
            self.ip = lambda q1,q2: (q1*q2.conj()).real
            self.op = lambda q1,q2: q1*q2.conj()
        elif c101.ndim > 1 and c201.ndim == c101.ndim:
            self.ndim = c101.shape[1]
            self.ip = lambda q1,q2: (q1*q2.conj()).real.sum(axis=-1)
            self.op = lambda q1,q2: q1@q2.conj().transpose()
        else:
            raise ValueError("The SRVTs must provide curves of same dimension")

        self.resample(x1,x2)


    def resample(self,x1,x2):
        """
        Sets the grid and samples the SRVTs.

        Parameters
        ----------
        x1,x2 : ndarray
            Ordered sequences representing the grid.
        """

        self.x1 = x1
        self.x2 = x2
        self.shape = (self.x1.shape[0],self.x2.shape[0])

        # Compute sqrt(dx)
        sdx1 = np.sqrt(np.diff(x1))
        sdx2 = np.sqrt(np.diff(x1))
        if self.ndim == 1:
            sdx1 = sdx1[:,None]
            sdx2 = sdx2[:,None]

        # Compute q*sqrt(dx) and the lengths of the curves
        self.q1sdt = self.q1(x1[1:])*sdx1
        self.q2sdt = self.q2(x2[1:])*sdx2
        self.l1 = np.linalg.norm(self.q1sdt)**2
        self.l2 = np.linalg.norm(self.q2sdt)**2

        # Normalize qsdt
        if self.normalize:
            self.q1sdt /= np.sqrt(self.l1)
            self.q2sdt /= np.sqrt(self.l2)


    def __getitem__(self, i):
        """
        Returns fij * sqrt(dxi * dyj).
        """
        return np.maximum(self.ip(self.q1sdt[i[0]-1],self.q2sdt[i[1]-1]),0)


    def geodesic(self,tau,output='curve',length='invariant',shift=False):
        """
        Computes
        """

        # Specify the weighting function
        if self.unit_sphere:
            dist = self.distance()
            if dist==0:
                w = lambda t: t
            w = lambda t: np.sin(dist*t)/np.sin(dist)
        else:
            w = lambda t: t

        # Compute the srvts
        if self.ndim == 1:
            tau = tau[:,None]
            q = self.q1sdt[None,:]*w(1-tau) + self.q2sdt[None,:]*w(tau)
        else:
            tau = tau[:,None,None]
            q = self.q1sdt[None,:,:]*w(1-tau) + self.q2sdt[None,:,:]*w(tau)

        # Scale by length
        length = length.lower()
        if (length)=='invariant':
            l = 1
        elif (length)=='linear':
            l = (1-tau)*self.l1 + tau*self.l2
        elif (length)=='logarithmic':
            l = np.exp((1-tau)*np.log(self.l1) + tau*np.log(self.l2))
        else:
            raise ValueError("Unknown length type '" + length + "'")
        q *= np.sqrt(l)

        output = output.lower()
        if (output)=='srvt':
            return q
        elif (output)=='curve':
            if self.ndim == 1:
                c = np.cumsum(q*np.sqrt(self.ip(q,q)),axis=1)
                if shift is not False:
                    width = c.real.ptp(axis=1)
                    width += shift*width.mean()
                    c += width.cumsum()[:,None]
            else:
                c = np.cumsum(q*np.sqrt(self.ip(q,q))[:,:,None],axis=1)
            return c
        else:
            raise ValueError("Unknown output type '" + output + "'")

    def inner_product(self):
        return self.objective()


    def distance(self):
        """
        Computes the pre-shape space distance between the curves.
        """
        if self.unit_sphere:
            return np.arccos(min(1,self.inner_product()))
        if self.normalize:
            return np.sqrt(2-2*self.inner_product())
        return np.sqrt(self.l1+self.l2-2*self.inner_product())








class HJBSolverC(HJBSolverQ):
    """
    Class for solving the HJB equation when point evaluations of the curves are
    available. In this case we estimate fh = <sqrt(c1(x+h)-c1(x)), sqrt(c2(y+h)-c2(y))>
    """


    def __init__(self,c1,c2,x1,x2=None,scheme=VInf,unit_sphere=False,normalize=False):
        """
        Initializes a solver for the HJB equation based on SRVTs of the curves.

        Parameters
        scheme : Scheme
            The scheme used by the solver
        c1, c2 : callable
            Functions representing the curves. The functions accepts an
            ndarray (n,) and return either a real or complex ndarray (n,) or a
            real or complex ndarray (n,d).
        x1, x2 : ndarray (n,) or int
            Grids used for solving the HJB equation. If x1 is int, then the
            a uniform grid with x1+1 points will be used. Similar for x2.
        unit_sphere : bool
            If True, we use the unit sphere topology to compute geodesics.
            If False, we use the L2 topology.
        normalize : bool
            If True, the curves are normalized to be approximately unit length.
        """

        if isinstance(x1, numbers.Integral):
            x1 = np.linspace(0,1,x1+1)
        if x2 is None:
            x2 = x1
        elif isinstance(x2, numbers.Integral):
            x2 = np.linspace(0,1,x2+1)

        self.c1 = c1
        self.c2 = c2
        self.scheme = scheme
        self.unit_sphere = unit_sphere
        self.normalize = normalize

        # Find the dimension of the curves
        c101 = c1(np.array([0,1]))
        c201 = c2(np.array([0,1]))

        # Specify the according inner product
        if c101.ndim == 1 and c201.ndim == 1:
            self.ndim = 1
            self.ip = lambda q1,q2: (q1*q2.conj()).real
            self.op = lambda q1,q2: q1*q2.conj()
        elif c101.ndim > 1 and c201.ndim == c101.ndim:
            self.ndim = c101.shape[1]
            self.ip = lambda q1,q2: (q1*q2.conj()).real.sum(axis=-1)
            self.op = lambda q1,q2: q1@q2.conj().transpose()
        else:
            raise ValueError("The SRVTs must provide curves of same dimension")

        self.resample(x1,x2)


    def resample(self,x1,x2):
        """
        Samples the curves at the provided grids x1,x2 and computes the
        estimates qsdt = sqrt(c(xi+1)-c(xi)) used as approximations of fh.

        Parameters
        ----------
        x1,x2 : ndarray
            Ordered sequences representing the grid.
        """

        # Set the new grids
        self.x1 = x1
        self.x2 = x2
        self.shape = (self.x1.shape[0],self.x2.shape[0])

        # Compute q1sdt
        v = np.diff(self.c1(x1),axis=0)
        vv = self.ip(v,v)
        vv[vv==0] = 1
        self.q1sdt = v*vv**-.25 if self.ndim==1 else v*vv[:,None]**-.25

        # Compute q1sdt
        v = np.diff(self.c2(x2),axis=0)
        vv = self.ip(v,v)
        vv[vv==0] = 1
        self.q2sdt = v*vv**-.25 if self.ndim==1 else v*vv[:,None]**-.25

        self.l1 = np.linalg.norm(self.q1sdt)**2
        self.l2 = np.linalg.norm(self.q2sdt)**2

        if self.normalize:
            self.q1sdt /= np.sqrt(self.l1)
            self.q2sdt /= np.sqrt(self.l2)
