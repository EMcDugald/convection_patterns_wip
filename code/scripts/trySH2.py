import numpy as np
import mySH_v1 as mySh
import matplotlib.pyplot as plt
import utils

x = np.linspace(-16,16,256)
y = np.linspace(-16,16,256)
X,Y = np.meshgrid(x,y)
w0 = .1*(np.cos(X)+np.sin(Y))
dt = .1
R = 5*(-1./(1.+np.exp(-(X**2+2*Y**2-220.)))+1.)
L=x[len(x)-1]-x[0]

nSteps = 100
W10 = mySh.integrateSH(w0,R,dt,nSteps,L)
fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
ax1.imshow(W10)
plt.savefig("/Users/edwardmcdugald/Research/convection_patterns/code/figs/mySH_tst3.pdf")
utils.ndwrite(W10,"/Users/edwardmcdugald/Research/convection_patterns/code/data/W10_2.d")

nSteps = 1000
W100 = mySh.integrateSH(w0,R,dt,nSteps,L)
fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
ax2.imshow(W100)
plt.savefig("/Users/edwardmcdugald/Research/convection_patterns/code/figs/mySH_tst4.pdf")
utils.ndwrite(W100,"/Users/edwardmcdugald/Research/convection_patterns/code/data/W100_2.d")