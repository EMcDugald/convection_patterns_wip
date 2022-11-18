
%% Swift-Hohenberg equation

% We use a pseudospectral discretization along with Exponential
% time-differencing/RK4 to solve the Swift-Hohenberg equation on
% "arbitrary" domains.

% The equation we are solving is 
%   u_t = -(\Delta + 1)^2 u + R u - u^3

% To use a spectral discretization, we fit the domain inside a rectangle
% [0,L_x] X [0, L_y] and extend periodically. Since the patterns only form
% for R > 0, we choose R = -0.5 + R \chi_{\Omega}, where \chi is a smoothed
% characteristic function of \Omega. 

% To prevent aliasing due to the nonlinearity u^3, we need to keep twice as
% many sample points in each direction as the number of useful (i.e.
% low-frequency) modes. We can keep R "sharp" on the scale of the grid
% spacing, although we will typically use a low pass filtered version of R
% since it shouldn't play a big role in determining the pattern, and a
% "softer" boundary seems more robust.

% This idea borrows from "Global description of patterns far from onset: a
% case study" by Ercolani, Indik, Newell and Passot for the setup of
% Swift-Hohenberg, and from "Fourth-order time-stepping for stiff PDEs" by
% Kassam and Trefethen for the implementation of the ETD.

%%  Spatial   grid   and   initial   condition:

Lx = 25*pi;    % Size of enclosing periodic rectangle
Ly = 25*pi;
% we want N ~ L_x because max wavenumber is like k_x ~ pi*N/Lx which we
% want to be about 3. Likewise for Ly.

Nx = 256;
Ny = 256;
% Number of grid points in x and y. We will keep them as powers of 2.
% Useful number of modes = Nx Ny/4 

beta = 0.45; % Less than 0.5, sets relative size of the ellipse in the box
amplitude = 0.1; % for initial pattern. want this "small" compared to sqrt(R)

xx = Lx*(-Nx/2+1:Nx/2)'/Nx;
yy = Ly*(-Ny/2+1:Ny/2)'/Ny;
[X,Y] = ndgrid(xx,yy);

%R = 0.5*tanh(sqrt(Lx^2+Ly^2)*(beta-sqrt((X/Lx).^2+(Y/Ly).^2))/2);
R=.5*ones(Nx,Ny);
  
init_flag=1;
if (init_flag == 1)
    u0 = randn(Nx,Ny);
    u0 = amplitude*u0/norm(u0, inf);
elseif (init_flag == 2)
    u0 = amplitude*sin(Y);
elseif (init_flag == 3)    
    u0 = ellipse_init(X,Y,beta*Lx,beta*Ly,amplitude);
end

%%   Precompute   various   ETDRK4   scalar   quantities:

h = .5;   %   time   step

kx = 2*pi*[0:Nx/2 -Nx/2+1:-1]'/Lx;   %   wave   numbers
ky = 2*pi*[0:Ny/2 -Ny/2+1:-1]'/Ly;

[xi,eta]=ndgrid(kx,ky);

L = -(1-xi.^2-eta.^2).^2;   %   Fourier   multipliers
E = exp(h*L);   
E2= exp(h*L/2);

M = 16;   %   no.   of   points   for   complex   means
r = exp(1i*pi*((1:M)-.5)/M);   %   roots   of   unity. 
% This circle should suffice since max eigenvalue of Lh is 0, at k=0, Lh =
% -1/8

L2 = L(:); % convert to a single column

LR = h*L2(:,ones(M,1)) + r(ones(Nx*Ny,1),:); % adding r(j) to the jth column
Q = h*real(mean((exp(LR/2)-1)./LR,2));  % means in the 2 direction.
f1 = h*real(mean((-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3,2));
f2 = h*real(mean((2+LR+exp(LR).*(-2+LR))./LR.^3,2));
f3 = h*real(mean((-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3,2));

f1=reshape(f1,Nx,Ny);  % convert from columns to N by N matrices
f2=reshape(f2,Nx,Ny); 
f3=reshape(f3,Nx,Ny);
Q=reshape(Q,Nx,Ny); 

%clear LR;

%% Dealiasing

Fx = false(Nx,1); % Fx = 1 for high frequencies which will be set to zero
Fy = false(Ny,1); 

Fx((Nx/2+1-round(Nx/4):1+ Nx/2+round(Nx/4)))= true; % For dealiasing with u^3
Fy((Ny/2+1-round(Ny/4):1+ Ny/2+round(Ny/4)))= true;

[alxi,aleta] = ndgrid(Fx,Fy); 
ind = alxi | aleta; %de-aliasing index.

%filter R and u0

Rhat = fftn(R);
Rhat(ind) = 0;
R = real(ifftn(Rhat));

vv = fftn(u0);
vv(ind) = 0;
u0 = real(ifftn(vv));

Q(ind) = 0; % Q is the only term that multiplies the nonlinear factors

%%   Main   time-stepping   loop:

tmax   =   100;   

nmax   =   round(tmax/h);   
nplt   =   floor((tmax/100));

tt = zeros(1,round(nmax/nplt)+1);
uu   =   zeros(Nx,Ny,round(nmax/nplt)+1);
tt(1) = 0;
uu(:,:,1) = u0;

% vv = fftn(u0);

ii = 1;

tic

for n   =   1:nmax
    t   =   n*h;
    
    Nv   =  fftn(R.*u0 - u0.^3);
    
    a   =   E2.*vv   +   Q.*Nv;
%    a(ind) = 0;
    ua  =   real(ifftn(a));
    Na   =  fftn(R.*ua - ua.^3);
    
    b   =   E2.*vv   +   Q.*Na; 
%    b(ind) = 0;
    ub  =   real(ifftn(b));
    Nb   =  fftn(R.*ub - ub.^3);
    
    c   =   E2.*a   +   Q.*(2*Nb-Nv);
%    c(ind) = 0;
    uc  =   real(ifftn(c));
    Nc   =  fftn(R.*uc - uc.^3);
    
    vv   =   E.*vv   +   Nv.*f1   +   2*(Na+Nb).*f2   +   Nc.*f3;
%     vv(ind) = 0;
    u0  =   real(ifftn(vv));
    
    if   mod(n,nplt)==0
      
        uu(:,:,ii+1)   =   u0;
        tt(ii+1)   =  t;
        ii = ii+1;
    end
end

toc

figure(1)
clf;
surf(X,Y,u0,'mesh','none');
axis equal;
view(2);

save('sh2.mat','tt','xx','yy','uu')
