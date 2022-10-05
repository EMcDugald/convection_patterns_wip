import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp

def reaction_diffusion_rhs(t,uvt,K22,d1,d2,beta,n,N):
    ut = np.reshape(uvt[0:N-1],(n,n))
    vt = np.reshape(uvt[N:2*N-1],(n,n))
    u = np.real(sp.fft.ifft2(ut))
    v = np.real(sp.fft.ifft2(vt))
    u3 = u**3
    v3 = v**3
    u2v = (u**2)*v
    uv2 = u*(v**2)
    utrhs = np.reshape(sp.fft.fft2(u-u3-uv2+beta*u2v+beta*v3),(N,1))
    vtrhs = np.reshape(sp.fft.fft2(v-u2v-v3-beta*u3-beta*uv2),(N,1))
    return np.concatenate((-d1*K22*uvt[0:N-1]+utrhs,-d2*K22*uvt[N:2*N-1]+vtrhs))

t = np.arange(0,500.05,.05)
d1=0.1
d2=0.1
beta=1.0
L=20
n=100
N=n**2
x2 = np.linspace(-L/2,L/2,n+1)
x = x2[0:len(x2)-1]
y=x
kx = (2.*np.pi/L)*sp.fft.fftfreq(n,1./n)
ky=kx
X, Y = np.meshgrid(x,y)
KX, KY = np.meshgrid(kx,ky)
K2 = KX**2+KY**2
K22 = np.reshape(K2,(N,1))
m=1.0
f=np.exp(-.01*(X**2+Y**2))

u = np.zeros(shape=(len(x),len(y),len(t)))
v = np.zeros(shape=(len(x),len(y),len(t)))
uf = np.zeros(shape=(len(x),len(y),len(t)))
vf = np.zeros(shape=(len(x),len(y),len(t)))

du = np.zeros(shape=(len(x),len(y),len(t)))
dv = np.zeros(shape=(len(x),len(y),len(t)))
duf = np.zeros(shape=(len(x),len(y),len(t)))
dvf = np.zeros(shape=(len(x),len(y),len(t)))

u[:,:,0]=np.tanh(np.sqrt(X**2+Y**2))*np.cos(m*np.angle(X+Y*1j)-np.sqrt(X**2+Y**2))
v[:,:,0]=np.tanh(np.sqrt(X**2+Y**2))*np.sin(m*np.angle(X+Y*1j)-np.sqrt(X**2+Y**2))
uf[:,:,0]=f*u[:,:,0]
vf[:,:,0]=f*v[:,:,0]

uvt = np.hstack((np.reshape(sp.fft.fft2(u[:,:,0]),(1,N)),np.reshape(sp.fft.fft2(v[:,:,0]),(1,N)))).T
uvt_rhs = reaction_diffusion_rhs(t[0],uvt,K22,d1,d2,beta,n,N)
du[:,:,0]=np.real(sp.fft.ifft2(np.reshape(uvt_rhs[0:N-1].T,(n,n))))
dv[:,:,0]=np.real(sp.fft.ifft2(np.reshape(uvt_rhs[N,2*N-1].T,(n,n))))

uvsol = solve_ivp(reaction_diffusion_rhs,t,uvt,K22,d1,d2,beta,n,N)



















