import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import copy

from srvpy.open.schemes import *
from srvpy.open.hjb import *
import srvpy.open.samplepaths as sp













c1 = sp.Curve(sp.PAWN,unit_length=True)
c2 = sp.Curve(sp.QUEEN,unit_length=True)

n = 5000
t = np.linspace(0,1,n+1)
c1 = interp1d(t,c1(t))
c2 = interp1d(t,c2(t))


n = 2000
solver = HJBSolverC(VInf,c1,c2,n,n)
solver.solve_hjb()
phi = solver._backtrack_hjb_constL1()
solver.resample(phi[:,0],phi[:,1])

for i in range(3):
    solver.solve_hjb_strip(10)
    phi = solver._backtrack_hjb_constL1()
    solver.resample(phi[:,0],phi[:,1])
u = solver.inner_product()
print(u)


w = 5


scheme = VInf


ns = (10**np.arange(1,2.9,0.25)).astype(int)
err = np.ones((ns.shape[0],4))
err2 = np.ones((ns.shape[0],3))
for i in range(ns.shape[0]):
    solver = HJBSolverC(VInf,c1,c2,ns[i],ns[i])
    solver.solve_hjb()

    phi = solver._backtrack_hjb_constL1()
    solver.resample(phi[:,0],phi[:,1])
    err[i,0] = abs(u-solver.inner_product())


    phi1 = phi[:,0]
    phi2 = phi[:,1]
    solver2 = HJBSolverC(VInf,
                         c1,
                         c2,
                         interp1d(phi1+phi2,phi1)(np.linspace(0,2,2*ns[i]+1)),
                          interp1d(phi1+phi2,phi2)(np.linspace(0,2,2*ns[i]+1)))
    solver2.solve_hjb_strip(2*w)
    phi = solver2._backtrack_hjb_constL1()
    solver2.resample(phi[:,0],phi[:,1])
    err2[i,0] = abs(u-solver2.inner_product())
    solver2 = HJBSolverC(VInf,
                         c1,
                         c2,
                         interp1d(phi1+phi2,phi1)(np.linspace(0,2,4*ns[i]+1)),
                         interp1d(phi1+phi2,phi2)(np.linspace(0,2,4*ns[i]+1)))
    solver2.solve_hjb_strip(4*w)
    phi = solver2._backtrack_hjb_constL1()
    solver2.resample(phi[:,0],phi[:,1])
    err2[i,1] = abs(u-solver2.inner_product())
    solver2 = HJBSolverC(VInf,
                         c1,
                         c2,
                         interp1d(phi1+phi2,phi1)(np.linspace(0,2,8*ns[i]+1)),
                         interp1d(phi1+phi2,phi2)(np.linspace(0,2,8*ns[i]+1)))
    solver2.solve_hjb_strip(8*w)
    phi = solver2._backtrack_hjb_constL1()
    solver2.resample(phi[:,0],phi[:,1])
    err2[i,2] = abs(u-solver2.inner_product())



    solver.solve_hjb_strip(w)
    phi = solver._backtrack_hjb_constL1()
    solver.resample(phi[:,0],phi[:,1])
    err[i,1] = abs(u-solver.inner_product())

    solver.solve_hjb_strip(w)
    phi = solver._backtrack_hjb_constL1()
    solver.resample(phi[:,0],phi[:,1])
    err[i,2] = abs(u-solver.inner_product())

    solver.solve_hjb_strip(w)
    phi = solver._backtrack_hjb_constL1()
    solver.resample(phi[:,0],phi[:,1])
    err[i,3] = abs(u-solver.inner_product())

    print(ns[i])


plt.plot(ns,err[:,0],'C0')
plt.plot(2*ns,err[:,1],'C1')
plt.plot(4*ns,err[:,2],'C2')
plt.plot(8*ns,err[:,3],'C3')
plt.plot(2*ns,err2[:,0],'C1--')
plt.plot(4*ns,err2[:,1],'C2--')
plt.plot(8*ns,err2[:,2],'C3--')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.show()
