function u0 = ellipse_init(X,Y,a,b,amp)
%ELLIPSE_INIT: Generates Dirichlet initial conditions based on the distance
%to the boundary of an ellipse.

nmx = 256;
q = 2*pi*(1:nmx)/nmx;

[imx, jmx] = size(X);

bdry = [a*cos(q);b*sin(q)];

rho = zeros(imx,jmx);

for ii = 1:imx
    for jj = 1:jmx
        rho(ii,jj) = min((X(ii,jj)-bdry(1,:)).^2+...
            (Y(ii,jj)-bdry(2,:)).^2);
    end
end

% filter 
kx = pi*[0:imx/2 -imx/2+1:-1]'/a;   %   wave   numbers
ky = pi*[0:jmx/2 -jmx/2+1:-1]'/b;

[xi,eta]=ndgrid(kx,ky);

rho = ifftn(exp(-(xi.^2+eta.^2)).*fftn(rho));

u0 = amp*sin(sqrt(rho));

% surf(X,Y,u0,'mesh','none');
% axis equal;
% view([0 0 30]);
end

