import numpy as np
import scipy as sp
from numpy import linalg as LA
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
import itertools
import operator
from scipy.fft import fft2, ifft2, fftfreq


def PolyDiffPoint(u, x, deg=3, diff=1):
    """
    Fits a chebyshev polynomial to the data, and
    takes the derivative. Using this for u_t estimates
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    """
    n = len(x)
    index = (n - 1) // 2
    # Fit to a polynomial
    poly = np.polynomial.chebyshev.Chebyshev.fit(x, u, deg)
    # Take derivative
    return poly.deriv(m=1)(x[index])

def BackwardDiff(u_curr, u_past, dt):
    """
    :param u_curr: current time func values
    :param u_past: previous time func values
    :param dt: time step
    :return: backward finite diff approximation
    """
    return (u_curr - u_past)/dt


def SpectralDerivs(func,Lx,Ly,type='x'):
    """
    :param func: function values
    :param Lx: length of rectangle in x dir
    :param Ly: length of rectangle in y dir
    :param type: derivative type: 'x', 'xx',
    'y', 'yy', 'xy', 'xxyy', 'xxxx', 'yyyy',
    'laplacian', 'biharmonic'
    :return: grid of derivative values
    """
    ny, nx = np.shape(func)
    kx = (2.*np.pi/Lx)*fftfreq(nx,1./nx)
    ky = (2.*np.pi/Ly)*fftfreq(ny,1./ny)
    Kx, Ky = np.meshgrid(kx, ky)
    if type == 'x':
        return np.real(ifft2(1j*Kx*fft2(func)))
    elif type == 'xx':
        return np.real(ifft2((1j*Kx)**2*fft2(func)))
    elif type == 'y':
        return np.real(ifft2(1j*Ky*fft2(func)))
    elif type == 'yy':
        return np.real(ifft2((1j*Ky)**2*fft2(func)))
    elif type == 'xy':
        return np.real(ifft2((1j*Ky)*(1j*Kx)*fft2(func)))
    elif type == 'xxyy':
        return np.real(ifft2((1j*Ky)**2*(1j*Kx)**2*fft2(func)))
    elif type == 'xxxx':
        return np.real(ifft2((1j*Kx)**4*fft2(func)))
    elif type == 'yyyy':
        return np.real(ifft2((1j*Ky)**4*fft2(func)))
    elif type == 'laplacian':
        fourierLaplacian = -(Kx**2+Ky**2)
        return np.real(ifft2(fourierLaplacian*fft2(func)))
    elif type == 'biharmonic':
        fourierLaplacian = -(Kx**2+Ky**2)
        fourierBiharm = fourierLaplacian*fourierLaplacian
        return np.real(ifft2(fourierBiharm*fft2(func)))
    else:
        raise Exception("Incompatible type selection")


def print_pde(w, rhs_description, ut='u_t'):
    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            pde = pde + "(%05f %+05fi)" % (w[i].real, w[i].imag) + rhs_description[i] + "\n   "
            first = False
    print(pde)


def TrainSTRidge(R, Ut, lam, d_tol, maxit=25, STR_iters=10, l0_penalty=None, normalize=2, split=0.8,
                 print_best_tol=False):
    """
    This function trains a predictor using STRidge.
    It runs over different values of tolerance and trains predictors on a training set, then evaluates them
    using a loss function on a holdout set.
    Please note published article has typo.  Loss function used here for model selection evaluates fidelity using 2-norm,
    not squared 2-norm.
    """

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0)  # for consistancy
    n, _ = R.shape
    train = np.random.choice(n, int(n * split), replace=False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train, :]
    TestR = R[test, :]
    TrainY = Ut[train, :]
    TestY = Ut[test, :]
    D = TrainR.shape[1]

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty == None: l0_penalty = 0.001 * np.linalg.cond(R)

    # Get the standard least squares estimator
    w = np.zeros((D, 1))
    w_best = np.linalg.lstsq(TrainR, TrainY, rcond=None)[0]
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty * np.count_nonzero(w_best)
    tol_best = 0

    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(TrainR, TrainY, lam, STR_iters, tol, normalize=normalize)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty * np.count_nonzero(w)

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol

        else:
            tol = max([0, tol - 2 * d_tol])
            d_tol = 2 * d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol: print("Optimal tolerance:", tol_best)

    return w_best


def STRidge(X0, y, lam, maxit, tol, normalize=2, print_results=False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.
    This assumes y is only one column
    """

    n, d = X0.shape
    X = np.zeros((n, d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0

    # Get the standard ridge esitmate
    if lam != 0:
        w = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y), rcond=None)[0]
    else:
        w = np.linalg.lstsq(X, y, rcond=None)[0]
    num_relevant = d
    biginds = np.where(abs(w) > tol)[0]

    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where(abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]

        # If nothing changes then stop
        if num_relevant == len(new_biginds):
            break
        else:
            num_relevant = len(new_biginds)

        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0:
                # if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else:
                break
        biginds = new_biginds

        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0:
            w[biginds] = \
            np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)), X[:, biginds].T.dot(y),
                            rcond=None)[0]
        else:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

    if normalize != 0:
        return np.multiply(Mreg, w)
    else:
        return w

def douglas_rachford(time_der,features,lam,gamma=.5,mu=1.):
    coeffs = np.ones(len(features))
    H1 = lam*np.linalg.norm(coeffs,ord=1)
    H2 = .5*np.sum(np.linalg.norm(tim_der-features*ceoffs))
    prox_H1 = np.max(np.abs(coeffs)-gamma)
    return coeffs