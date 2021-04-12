import numpy as np
import matplotlib.pyplot as plt
from srvf.open import Registration
from srvf.open.schemes import VInf
from srvf.open.paths import HAND1, HAND2
from svgpathtools import Path

path1 = Path(HAND1)
path2 = Path(HAND2)

n = 1001
t = np.linspace(0,1,n+1)
c1 = np.array([path1.point(t) for t in t])
c2 = np.array([path2.point(t) for t in t])
l1 = path1.length()
l2 = path2.length()


align = Registration(c1,c2,l1=l1,l2=l2)
align.setup(VInf)

tau = np.linspace(0,1,7)
gamma = align.preshapeGeodesic(tau)
gamma += 20*np.arange(7)[None,:]
for i in range(7):
    plt.plot(gamma[:,i].real,gamma[:,i].imag)
plt.show()

align.setup(VInf)
align.register()

tau = np.linspace(0,1,7)
gamma = align.preshapeGeodesic(tau)
gamma += 20*np.arange(7)[None,:]

for i in range(7):
    plt.plot(gamma[:,i].real,gamma[:,i].imag)
plt.show()
