import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import copy

from srvpy.open.schemes import *
from srvpy.open.hjb import *
import srvpy.open.samplepaths as sp



c1 = sp.Curve(sp.HAND3,unit_length=True)
c2 = sp.Curve(sp.HAND3,unit_length=True)
n = 5*2**10
t = np.linspace(0,1,n+1)
c1 = interp1d(t,c1(t))
c2 = interp1d(t,c2(t))

phi1 = lambda t: t+0.7*np.sin(3*np.pi*t)/(3*np.pi)
phi2 = lambda t: t-0.6*np.sin(2*np.pi*t)/(2*np.pi)


def qsdt(c,phi):
    v = np.diff(c(phi))/np.diff(phi)
    vabs = np.abs(v)
    vabs[vabs==0] = 1
    return v/np.sqrt(vabs)

n = 5*2**10
t = np.linspace(0,1,n+1)
j = n*(qsdt(c1,phi1(t))*qsdt(c2,phi2(t)).conj()).real*np.sqrt(np.diff(phi1(t))*np.diff(phi2(t)))

for m in [1,2,3,4,5,6,7,8,9]:

    n = 5*2**m
    t = np.linspace(0,1,n+1)
    di = 2**(10-m)
    j2 = n*(qsdt(c1,phi1(t))*qsdt(c2,phi2(t)).conj()).real*np.sqrt(np.diff(phi1(t))*np.diff(phi2(t)))
    j3 = n*(np.diff(c1(phi1(t)))*np.diff(c2(phi2(t))).conj()).real/np.sqrt(np.diff(phi1(t))*np.diff(phi2(t)))
    print(np.abs(j[::di]-j2).max(),
          np.abs(j[::di]-j3).max())
    plt.plot(j[::di])
    plt.plot(j2)
    plt.plot(j3)
    plt.show()
