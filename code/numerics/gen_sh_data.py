import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt

#-- Set geometry and initial conditions --#
#Lx = 25*np.pi
#Ly = 25*np.pi
Lx = 80*np.pi
Ly = 20*np.pi
#Nx = 256
#Ny = 256
Nx = 1024
Ny = 256
beta = .40 #less than .5, sets relative size of ellipse in box
amplitude = 0.1 #for initial pattern. want this "small" compared to sqrt(R)
xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
#ToDo: clear up confusion regarding cartesian vs matrix indexing
X,Y = np.meshgrid(xx,yy,indexing='ij')
#X,Y = np.meshgrid(xx,yy)

# Use the below R for solution on Ellipse
R = .5*np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)

# Use the below R for solution on square
#R = .5*np.ones((Nx,Ny))

# method to be called for setting initial condition for solution on ellipse
def ellipse_init(X,Y,a,b,amp):
    nmx = 256
    q = 2*np.pi*np.arange(1,nmx+1,1)/nmx
    imx, jmx = np.shape(X)
    bdry = np.vstack((a*np.cos(q), b*np.sin(q)))
    rho = np.zeros((imx,jmx))
    for ii in range(imx):
        for jj in range(jmx):
            rho[ii,jj] = np.min((X[ii,jj]-bdry[0,:])**2+(Y[ii,jj]-bdry[1,:])**2)
    kx = (np.pi/a)*sp.fft.fftfreq(imx,1./imx)
    ky = (np.pi/b)*sp.fft.fftfreq(jmx,1./jmx)
    xi,eta = np.meshgrid(kx,ky,indexing='ij')
    rho = sp.fft.ifft2(np.exp(-(xi**2+eta**2))*sp.fft.fft2(rho))
    u0 = np.real(amp*np.sin(np.sqrt(rho)))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 2))
    ax.imshow(u0.T)  # show transpose to convert from matrix indexing to cartesian indexing
    plt.show()
    return u0

#set init flag to determine initial condition
init_flag = 3
if init_flag==1:
    u0 = np.random.randn(Nx,Ny)
    u0 = amplitude*u0/np.linalg.norm(u0,np.inf)
elif init_flag==2:
    u0 = amplitude*np.sin(Y)
else:
    u0 = ellipse_init(X,Y,beta*Lx,beta*Ly,amplitude)


#-- precompute ETDRK4 scalar quantities --#
h=.01 #time step
kx = (2.*np.pi/Lx)*sp.fft.fftfreq(Nx,1./Nx) #wave numbers
ky = (2.*np.pi/Ly)*sp.fft.fftfreq(Ny,1./Ny)
#ToDo: clear up confusion regarding cartesian vs matrix indexing
xi, eta = np.meshgrid(kx,ky,indexing='ij')
#xi, eta = np.meshgrid(kx,ky)
L = -(1-xi**2-eta**2)**2
E = np.exp(h*L)
E2 = np.exp(h*L/2)

M=16 # number of points for complex means
r = np.exp(1j*np.pi*((np.arange(1,M+1,1)-.5)/M)) #roots of unity
L2 = L.flatten() #convert to single column
LR = h*np.vstack([L2]*M).T + np.vstack([r]*Nx*Ny) #adding r(j) to jth column
Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR,1)) #means in the 2 directions
f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3,1))
f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3,1))
f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3,1))

f1 = np.reshape(f1,(Nx,Ny))
f2 = np.reshape(f2,(Nx,Ny))
f3 = np.reshape(f3,(Nx,Ny))
Q = np.reshape(Q,(Nx,Ny))

#dealiasing
Fx = np.zeros((Nx,1),dtype=bool) #Fx = 1 for high frequencies which will be set to 0
Fy = np.zeros((Ny,1),dtype=bool)
Fx[int(Nx/2-np.round(Nx/4)):int(1+ Nx/2+np.round(Nx/4))] = True
Fy[int(Ny/2-np.round(Ny/4)):int(1+ Ny/2+np.round(Ny/4))] = True

alxi, aleta = np.meshgrid(Fx,Fy,indexing='ij')
#alxi, aleta = np.meshgrid(Fx,Fy)
ind = alxi | aleta #de-aliasing index

#filter R an u0

Rhat = sp.fft.fft2(R)
Rhat[ind] = 0
R = np.real(sp.fft.ifft2(Rhat))

vv = sp.fft.fft2(u0)
vv[ind] = 0
u0 = np.real(sp.fft.ifft2(vv))

Q[ind] = 0 #Q is the only term the multiplies the non linear factors

tmax = 5
nmax = np.round(tmax/h)
plt_fac = 5 #set this to tmax if you want every time step saved
nplt = np.floor(tmax/plt_fac)

tt = np.zeros(int(np.round(nmax/nplt)+1))
uu = np.zeros((Nx,Ny,int(np.round(nmax/nplt)+1)))
uu[:,:,0] = u0
tt[0] = 0

ii=0
start = time.time()

for n in range(1,int(nmax)+1):
    t = n*h
    Nv = sp.fft.fft2(R*u0-u0**3)
    a = E2*vv + Q*Nv
    ua = np.real(sp.fft.ifft2(a))
    Na = sp.fft.fft2(R*ua - ua**3)
    b = E2*vv + Q*Na
    ub = np.real(sp.fft.ifft2(b))
    Nb = sp.fft.fft2(R*ub - ub**3)
    c = E2*a + Q*(2*Nb-Nv)
    uc = np.real(sp.fft.ifft2(c))
    Nc = sp.fft.fft2(R*uc - uc**3)
    vv = E*vv + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    u0 = np.real(sp.fft.ifft2(vv))

    if n%nplt==0:
        uu[:,:,ii+1]=u0
        tt[ii+1]=t
        ii = ii+1

end = time.time()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,2))
ax.imshow(u0.T) #show transpose to convert from matrix indexing to cartesian indexing
plt.show()

mdict={"tt":tt.reshape(1,len(tt)),"xx":xx.reshape(Nx,1),"yy":yy.reshape(Ny,1),"uu":uu}

#Ran this with init_flag=1, Nx,Ny=256, Lx,Ly=25pi, R=.5, tmax = 100, h=.5, beta=.45
#sp.io.savemat("/Users/edwardmcdugald/Research/convection_patterns_wip/code/data/SHpy1.mat",mdict)

#Ran this with init_flag=2, Nx,Ny=256, Lx,Ly=25pi, R=.5, tmax = 100, h = .5, beta=.45
#sp.io.savemat("/Users/edwardmcdugald/Research/convection_patterns_wip/code/data/SHpy2.mat",mdict)
#matlab version is /Users/edwardmcdugald/Research/convection_patterns_matlab/sh_mat2py_tst.mat

#Ran this with init_flag = 3, R=Tanh function, Nx=512, Ny=128, Lx = 80pi, Ly=20pi, tmax = 100, h=.5, beta=.4
#sp.io.savemat("/Users/edwardmcdugald/Research/convection_patterns_wip/code/data/SHpy3.mat",mdict)
#matlab version is /Users/edwardmcdugald/Research/convection_patterns_matlab/sh_mat2py_tst2.mat

#Ran this with init_flag = 3, R=Tanh function, Nx=512, Ny=128, Lx = 80pi, Ly=20pi, tmax = 10, h=.1, beta=.4
#sp.io.savemat("/Users/edwardmcdugald/Research/convection_patterns_wip/code/data/SHpy4.mat",mdict)

#Ran this with init_flag = 3, R=Tanh function, Nx=1024, Ny=256, Lx = 80pi, Ly=20pi, tmax = 5, h=.01, beta=.4
sp.io.savemat("/Users/edwardmcdugald/Research/convection_patterns_wip/code/data/SHpy5.mat",mdict)

print("time to generate solutions: ", end-start)











