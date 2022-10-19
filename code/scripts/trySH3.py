import numpy as np
import mySH_v2 as mySh
import matplotlib.pyplot as plt
import utils

x = np.linspace(-16,16,512)
y = np.linspace(-16,16,512)
X,Y = np.meshgrid(x,y)
w0 = .1*(np.cos(X)+np.sin(Y))
dt = .1
R = .5*(-1./(1.+np.exp(-(X**2+2*Y**2-256.)))+1.)
L=x[len(x)-1]-x[0]

nSteps = 100
W10 = mySh.integrateSH(w0,R,dt,nSteps,L)
fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
ax1.imshow(W10)
plt.savefig("/Users/edwardmcdugald/Research/convection_patterns/code/figs/trySH3_ncR_100.pdf")
utils.ndwrite(W10,"/Users/edwardmcdugald/Research/convection_patterns/code/data/trySH3_ncR_100.d")

nSteps = 1000
W100 = mySh.integrateSH(w0,R,dt,nSteps,L)
fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
ax2.imshow(W100)
plt.savefig("/Users/edwardmcdugald/Research/convection_patterns/code/figs/trySH3_ncR_1000.pdf")
utils.ndwrite(W100,"/Users/edwardmcdugald/Research/convection_patterns/code/data/trySH3_ncR_1000.d")

R=.5

nSteps = 100
W10 = mySh.integrateSH(w0,R,dt,nSteps,L)
fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
ax3.imshow(W10)
plt.savefig("/Users/edwardmcdugald/Research/convection_patterns/code/figs/trySH3_100.pdf")
utils.ndwrite(W10,"/Users/edwardmcdugald/Research/convection_patterns/code/data/trySH3_100.d")

nSteps = 1000
W100 = mySh.integrateSH(w0,R,dt,nSteps,L)
fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
ax4.imshow(W100)
plt.savefig("/Users/edwardmcdugald/Research/convection_patterns/code/figs/trySH3_1000.pdf")
utils.ndwrite(W100,"/Users/edwardmcdugald/Research/convection_patterns/code/data/trySH3_1000.d")