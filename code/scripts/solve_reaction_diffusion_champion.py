import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.io import savemat

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


def reaction_diffusion_rhs(t_dummy,uvt):
    uvt = np.reshape(uvt,(2*N,1))
    ut = np.reshape(uvt[0:N],(n,n))
    vt = np.reshape(uvt[N:2*N],(n,n))
    u = np.real(sp.fft.ifft2(ut.T))
    v = np.real(sp.fft.ifft2(vt.T))
    u3 = u**3
    v3 = v**3
    u2v = (u**2)*v
    uv2 = u*(v**2)
    utrhs = np.reshape(sp.fft.fft2((u-u3-uv2+beta*u2v+beta*v3).T),(N,1))
    vtrhs = np.reshape(sp.fft.fft2((v-u2v-v3-beta*u3-beta*uv2).T),(N,1))
    c1 = -d1*K22*uvt[0:N]+utrhs
    c2 = -d2*K22*uvt[N:2*N]+vtrhs
    return np.array([[c1,c2]]).reshape((2*N,1))


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

uvt = np.hstack((np.reshape(sp.fft.fft2(u[:,:,0].T),(1,N)),np.reshape(sp.fft.fft2(v[:,:,0].T),(1,N)))).T
uvt_rhs = reaction_diffusion_rhs(t[0],uvt)
du[:,:,0]=np.real(sp.fft.ifft2(np.reshape(uvt_rhs[0:N].T,(n,n))))
dv[:,:,0]=np.real(sp.fft.ifft2(np.reshape(uvt_rhs[N:2*N].T,(n,n))))

uvsol = np.zeros(shape=(N+1,2*N),dtype='complex')
uvsol[0,:]=uvt.flatten()
for i in range(1,len(t)):
    if i%1000==0:
        print(i)
    tspan = [t[i-1],t[i]]
    y0 = uvt.flatten()
    sol = solve_ivp(reaction_diffusion_rhs, tspan, y0,vectorized=True)
    uvsol[i, :] = sol.y[:,len(sol.t)-1]
    uvt = uvsol[i,:]

for j in range(len(t)-1):
    ut = np.reshape(uvsol[j,0:N].T,(n,n))
    vt = np.reshape(uvsol[j,N:2*N].T,(n,n))
    u[:,:,j+1]=np.real(sp.fft.ifft2(ut))
    v[:,:,j+1]=np.real(sp.fft.ifft2(vt))

    uvt_rhs = reaction_diffusion_rhs(t[j+1],uvsol[j,0:2*N].T)
    du[:,:,j+1] = np.real(sp.fft.ifft2(np.reshape(uvt_rhs[0:N].T,(n,n))))
    dv[:,:,j+1] = np.real(sp.fft.ifft2(np.reshape(uvt_rhs[N:2*N].T,(n,n))))

    uf[:,:,j+1] = f*u[:,:,j+1]
    vf[:,:,j+1] = f*v[:,:,j+1]
    duf[:,:,j+1] = f*du[:,:,j+1]
    dvf[:,:,j+1] = f*dv[:,:,j+1]

t = t[2:N+1]
uf = uf[:,:,2:N+1]
vf = vf[:,:,2:N+1]
duf = duf[:,:,2:N+1]
dvf = dvf[:,:,2:N+1]

mdict={"t":t,"x":x,"y":y,"uf":uf,"vf":vf,"duf":duf,"dvf":dvf}
savemat("/Users/edwardmcdugald/Research/convection_patterns/code/data/rd1.mat",mdict)

print("debug")























