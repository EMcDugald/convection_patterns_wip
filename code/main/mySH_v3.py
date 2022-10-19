import numpy as np
import scipy as sp

#SH Solver using RK4 for Nonlin part

def non_lin_rhs(w,R):
    return (R-1)*w - w**3

def integrateSH(w0,R,dt,nSteps,L):
     """
     :param w0: initial temperature surface
     :param R: bifurcation parameter- can be a constant, or of same shape as w0
     :param dt: time step length
     :param nSteps: number of time steps to take
     :param L: Length of square over which w0 is defined
     :return w0: time evolution of w0 at time 0+dt*nSteps
     Ideally, the size of w0 is fft friendly, ie 2^n x 2^n
     """
     print("Starting time integration of Swift Hohenberg")
     ny, nx = np.shape(w0)
     print("Dimensions of w0:", nx, ny)
     kx = (2.*np.pi/L)*sp.fft.fftfreq(nx,1./nx)
     ky = (2.*np.pi/L)*sp.fft.fftfreq(ny,1./ny)
     Kx, Ky = np.meshgrid(kx,ky)
     fourierLaplacian = -(Kx**2+Ky**2)
     A = -(fourierLaplacian*fourierLaplacian)-2*fourierLaplacian
     for i in range(0,nSteps):
         if i%100 == 0:
             print("step number:",i)
         w1 = np.real(sp.fft.ifft2(np.exp(A*.5*dt)*sp.fft.fft2(w0)))
         k1 = dt*w1
         k2 = dt*non_lin_rhs(w1+.5*k1, R)
         k3 = dt*non_lin_rhs(w1+.5*k2, R)
         k4 = dt*non_lin_rhs(w1+k3, R)
         w2 = (k1+2*k2+2*k3+k4)/6 + w1
         w0 = np.real(sp.fft.ifft2(np.exp(A*.5*dt)*sp.fft.fft2(w2)))
     return w0


