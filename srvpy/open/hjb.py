"""
This submodule contains objects for the numerical registration of curves and
computation of shape space geodesics in the Square Root Velocity Framework.
"""

# External dependencies:
import numpy as np
from math import gcd, ceil
from scipy.interpolate import interp1d

# Internal dependencies:
from .schemes import *


class Registration:
    """
    Object with necessary methods to register curves and compute geodesics in
    the Square Root Velocity Framework. The

    Attributes
    ----------
    c1, c2 : ndarray
        Point evaluations of the curves.
    q1, q2 : ndarray
        SRV representation of the curves.

    Methods
    -------
    setup(scheme,x1=None,x2=None)
        Computes the SRV representation of the curves.
    """

    def __init__(self,c1,c2,l1=None,l2=None,x1=None,x2=None):
        """
        Initializes the Registration object from point data.

        Parameters
        ----------
        c1, c2 : ndarray (n+1,) or ndarray (n+1,d)
            Point estimates of the curves. The arrays might have different
            length along the first dimension, but must have equal length along
            the second dimension. If the arrays are one-dimensional, the data
            type can be real or complex. Otherwise, the data type must be real.
        l1, l2 : float, optional
            Lengths of the curves. If not specified, they are estimated using
            linear interpolation of the points.
        x1, x2 : ndarray (n+1,), optional
            Grid points for the evlaluation of the curves.
        """

        # Store the grid points and the point evaluations
        self.c1 = c1
        self.c2 = c2
        self.l1 = l1
        self.l2 = l2
        self.shape = (c1.shape[0],c2.shape[0])
        self.setup(x1,x2)


    def setup(self,x1=None,x2=None):
        """
        Constructs interpolation objects for the curves, and computes their
        SRV representations.

        Parameters
        ----------
        x1, x2 : ndarray (n+1,), optional
            Grid points for the evlaluation of the curves. If not specified,
            a uniform grid is assumed.
        """
        self.x1 = x1 if x1 is not None else np.linspace(0,1,self.shape[0])
        self.x2 = x2 if x2 is not None else np.linspace(0,1,self.shape[1])

        # Construct interpolations of the curves
        self.c1f = interp1d(self.x1,self.c1)
        self.c2f = interp1d(self.x1,self.c2)

        # Construct the riemannian metric for the parameter space
        if self.c1.ndim == 1:
            self.ndim = 1
            self.ip = lambda q1,q2: (q1*q2.conj()).real
        else:
            self.ndim = self.c1.shape[1]
            self.ip = lambda q1,q2: (q1*q2).sum(axis=1)

        # Compute the SRV representations and estimate the lengths if not
        # specified
        self.computeSRV()


    def computeSRV(self):
        """
        Computes the SRV representation of the curves.
        """

        l1 = 1 if self.l1 is None else self.l1
        self.q1 = self.c2srv(self.c1,l1)
        if self.l1 is None:
            self.l1 = self.ip(q,q).sum()
            self.q1 /= np.sqrt(self.l1)

        l2 = 1 if self.l2 is None else self.l2
        self.q2 = self.c2srv(self.c2,l2)
        if self.l1 is None:
            self.l1 = self.ip(q,q).sum()
            self.q1 /= np.sqrt(self.l1)


    def c2srv(self,c,length):
        """
        Computes the square root velocity representation given point
        evaluations of the curve.

        Parameters
        ----------
        c : ndarray (n+1,) or ndarray (n+1,d)
            Point evaluations of a curve c.

        Returns
        -------
        q : ndarray (n,) or ndarray (n,d)
            The square root velocity representation q = dc/sqrt(l*|dc|).
        """
        return self.v2srv(np.diff(c),length)


    def v2srv(self,v,length):
        """
        Computes the square root velocity representation given point
        evaluations of the derivatives of the curve.

        Parameters
        ----------
        v : ndarray (n,) or ndarray (n,d)
            Point evaluations of the velocity of a curve.

        Returns
        -------
        q : ndarray (n,) or ndarray (n,d)
            The square root velocity representation q = v/sqrt(l*|v|).
        """
        vv = self.ip(v,v)
        vv[vv==0] = 1
        q = v*vv**-.25 if self.ndim==1 else v*vv[:,None]**-.25
        return q / np.sqrt(length)


    def __getitem__(self, i):
        """
        Returns fij * sqrt(dxi * dyj).
        """
        return np.maximum(self.ip(self.q1[i[0]-1],self.q2[i[1]-1]),0)


    def resample(self,phi):
        """
        Resamples the curves given a reparametrisation path phi.

        Parameters
        ---------
        phi : ndarray (k,2)
            Reparametrisation path.
        """

        # If phi is normalized, i.e. if phi is computed with respect to
        # the current grids x1,x2
        self.x1 = phi[:,0]
        self.x2 = phi[:,1]
        self.c1 = self.c1f(self.x1)
        self.c2 = self.c2f(self.x2)

        # Resample the srv representation
        self.q1 = self.c2srv(self.c1,self.l1)
        self.q2 = self.c2srv(self.c2,self.l2)

        self.u = None
        self.shape = (phi.shape[0],phi.shape[0])


    def inner_product(self):
        """
        Computes the inner product between the srv representations of the
        curves.
        """
        return self.ip(self.q1,self.q2).sum()


    def distance(self):
        """
        Computes the pre-shape space distance between the curves.
        """
        return np.arccos(min(1,self.inner_product()))


    def register(self,scheme=None):
        """
        Registrates the two curves. This method solves the HJB equation, then
        run the backtracking method to obtain the reparametrization path, and
        finally resamples the curves according to the backtracking procedure.

        Parameters
        scheme : Scheme, optional
            Changes the current scheme
        """
        self.solveHJB(scheme)
        phi = self.backtrackHJB()
        self.resample(phi)


    def solveHJB(self,scheme=None):
        """
        Approximates viscosity solutions of the Hamilton-Jacobi-Bellman for the
        value function.

        Parameters
        scheme : Scheme, optional
            Changes the current scheme
        """

        if scheme is not None:
            self.scheme = scheme

        # Initialize the value function
        self.u = np.zeros(self.shape)
        n = self.shape[0]-1
        m = self.shape[1]-1

        # Iterate through the anti-diagonals of the array.
        for k in range(2,n+m+1):
            i = np.arange(max(k-m,1), min(k,n+1))
            j = np.arange(max(k-n,1), min(k,m+1))[::-1]
            self.scheme.update(self.u,self,i,j)

        return self.u


    def backtrackHJB(self,phi1=None):
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

            a1,a2 = self.scheme.alpha(self.u,self,i,j)
            if a1==a2==0:
                a1 = 1

            west  = (self.x1[i], phi[k,1]-(phi[k,0]-self.x1[i])*a2/(a1+eps))
            south = (phi[k,0]-(phi[k,1]-self.x2[j])*a1/(a2+eps), self.x2[j])
            if west[0] <= south[0] or west[1] <= south[1]:
                phi[k-1] = south
            else:
                phi[k-1] = west

            if phi[k-1][0] == self.x1[i]:
                i = i-1
            if phi[k-1][1] == self.x2[j]:
                j = j-1

            continue

            b = h2*a1 - h1*a2
            if a2<=0 or phi[k,1]==0:
                phi[k-1] = i-1,phi[k,1]
            elif a1<=0 or phi[k,0]==0:
                phi[k-1] = phi[k,0],j-1
            elif b>=0:
                phi[k-1] = i-1, j-1 + b/a1
            else:
                phi[k-1] = i-1 - b/a2, j-1

        return phi


    def preshapeGeodesicSRV(self,tau):
        """
        Computes the srv representation of the preshape geodesic between two
        curves with respect to the current parametrization.

        Parameters
        ----------
        tau : ndarray
        """
        dist = np.arccos(min(1,self.ip(self.q1,self.q2).sum()))
        w1 = 1-tau if dist == 0 else np.sin(dist*(1-tau))/np.sin(dist)
        w2 =   tau if dist == 0 else np.sin(dist*   tau )/np.sin(dist)

        q = self.q1[:,None]*w1[None,:] + self.q2[:,None]*w2[None,:]
        return q


    def preshapeGeodesic(self,tau):
        """
        Computes the preshape geodesic between two curves with respect to the
        current parametrization.

        Parameters
        ----------
        tau : ndarray
        """

        # Pre-shape space geodesic
        q = self.preshapeGeodesicSRV(tau)
        c = np.cumsum(q*np.sqrt(self.ip(q,q)),axis=0)

        # Scale space geodesic
        length = np.exp((1-tau)*np.log(self.l1) + tau*np.log(self.l2))
        c *= length[None,:]

        # Rotation space geodesic


        # Translation space
        c0 = (1-tau)*self.c1[[0]] + tau*self.c2[[0]]
        c += c0[None,:]

        return c


class Registrationf(Registration):


    def __init__(self,c1f,c2f,l1=None,l2=None):
        """
        Initializes the Registration object from callables.

        Parameters
        ----------
        c1f, c2f : callables
            Callables representing the curves. The callables must have domain
            [0,1] and must allow for numpy input.
        """

        self.c1f = c1f
        self.c2f = c2f
        self.l1 = l1
        self.l2 = l2

    def setup(self,x1=None,x2=None,n=None,m=None):

        if x1 is None or x2 is None:
            assert n is not None, \
                "Either (x1, x2) or n (and m) must be specified."
            m = m if m is not None else n
            self.x1 = np.linspace(0,1,n+1)
            self.x2 = np.linspace(0,1,m+1)
            self.shape = (n+1,m+1)

        # Store the grid points and the point evaluations
        self.c1 = self.c1f(self.x1)
        self.c2 = self.c2f(self.x2)

        # Construct the riemannian metric for the parameter space
        if self.c1.ndim == 1:
            self.ndim = 1
            self.ip = lambda q1,q2: (q1*q2.conj()).real
        else:
            self.ndim = self.c1.shape[1]
            self.ip = lambda q1,q2: (q1*q2).sum(axis=1)


        # Compute the SRV representations and estimate the lengths if not
        # specified
        self.computeSRV()
