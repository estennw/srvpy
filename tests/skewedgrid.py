import numpy as np
import matplotlib.pyplot as plt

from srvpy.open.schemes import *
from srvpy.open.hjb import Registration
import srvpy.open.samplepaths as sp





m = 10
n = 5*2**m
t = np.linspace(0,1,n+1)

c1,l1 = sp.hand3(t)
c2,l1 = sp.hand5(t)

reg_ = Registration(c1,c2)
reg_.solve_hjb(VInf)


ms = np.arange(0,9)
schemes = [VInf,V12]
err = np.ones((len(schemes),ms.shape[0]))
for j in range(ms.shape[0]):
    for i in range(len(schemes)):
        di = 2**(m-ms[j])
        reg = Registration(c1[::di],c2[::di])
        reg.solve_hjb(schemes[i])
        err[i,j] = abs(np.sqrt(reg.value[-1,-1]) -
                       np.sqrt(reg_.value[-1,-1]))
        plt.subplot(1,len(schemes),i+1)
        plt.imshow(np.abs(reg_.value[::di,::di]-reg.value))
        plt.colorbar()
    plt.show()

plt.plot(5*2**ms,err[0])
plt.plot(5*2**ms,err[1])
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.show()
