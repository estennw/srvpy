import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import copy

from srvpy.open.schemes import *
from srvpy.open.hjb import *
import srvpy.open.samplepaths as sp





c1 = sp.Curve(sp.HAND3,unit_length=True)
c2 = sp.Curve(sp.HAND5,unit_length=True)



n = 500
solver = HJBSolverC(VInf,c1,c2,n,n)
solver.solve_hjb()
phi = solver.backtrack_hjb()
solver.resample(phi[:,0],phi[:,1])
"""
for i in range(3):
    solver.solve_hjb_strip(10)
    phi = solver._backtrack_hjb_constL1()
    solver.resample(phi[:,0],phi[:,1])
"""

def f(x1,x2):
    v1 = c1.v(x1)
    v2 = c2.v(x2)
    return np.maximum((v1*v2.conj()).real,0)**2/(np.abs(v1)*np.abs(v2))


phi1 = solver.x1[::16]
phi2 = solver.x2[::16]


dt = np.sqrt(np.diff(phi1)*np.diff(phi2))/np.sqrt(f(0.5*(phi1[1:]+phi1[:-1]),0.5*(phi2[1:]+phi2[:-1])))
t = np.r_[0,np.cumsum(dt)]

ddphi1 = (-phi1[:-2] + 2*phi1[1:-1] - phi1[2:])/(dt[1:]*dt[:-1])
ddphi2 = (-phi2[:-2] + 2*phi2[1:-1] - phi2[2:])/(dt[1:]*dt[:-1])

df1 = (f(phi1[2:],phi2[1:-1]) - f(phi1[:-2],phi2[1:-1]))/(phi1[2:]-phi1[:-2])
df2 = (f(phi1[1:-1],phi2[2:]) - f(phi1[1:-1],phi2[:-2]))/(phi2[2:]-phi2[:-2])

plt.plot(ddphi1)
plt.plot(df2)
plt.show()
