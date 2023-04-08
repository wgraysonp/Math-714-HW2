import numpy as np
import scipy.sparse as sp
import utils
import time


# the coefficient function a(r) = r^2 in PDE b(r) d/dr(a(r) u'(r)) = 0
# evaluated at the midpoints r_{i + 1/2}
def func_a(r):
    Nx = np.size(r)
    r_mid = 0.5*(r[:Nx-1] + r[1: Nx])
    a_mid = np.power(r_mid, 2)

    return a_mid


# returns b(r) = r^-2
def func_b(r):
    return np.power(r, -2)


# Returns operator matrix
def laplace_operator(Nx, dx, a, b):
    dxx = dx**2

    row = [0, 0, 0]
    col = [0, 1, 2]
    vals = [-3/(2*dx), 2/dx, -1/(2*dx)]

    for i in range(1, Nx-1):
        row.extend(3*[i])
        col.extend([i-1, i, i+1])
        vals.extend([a[i-1]*b[i]/dxx, -(a[i-1] + a[i])*b[i]/dxx, a[i]*b[i]/dxx])

    row.append(Nx-1)
    col.append(Nx-1)
    vals.append(1)

    A = sp.csr_matrix((vals, (row, col)), shape=(Nx, Nx))

    return A


# returns target vector accounting for boundary conditions u'(0.1) = alpha
# and u(10) = beta
def target(Nx,alpha, beta):
    F = np.zeros(Nx)
    F[0] = alpha
    F[-1] = beta

    return F


# returns analytic solution of ODE
def analytic(r):
    B = 0.2
    C = 25 + 0.2/10
    return (-B)*np.power(r, -1) + C


def main(Nx):
    # set boundary conditions
    alpha = 20
    beta = 25

    # create lists to store error and grid spacing
    E = []
    H = []

    for n in Nx:
        #define the grid
        r = np.linspace(0.1, 10, n)
        dr = r[1] - r[0]

        # define coefficient functions
        a = func_a(r)
        b = func_b(r)

        # get matrix and vector in equation AU = F
        F = target(n, alpha, beta)
        A = laplace_operator(n, dr, a, b)

        # solve the system
        U_comp = sp.linalg.spsolve(A, F)

        # define the analytic solution
        U_true = analytic(r)

        # and L-infinity error to E and spacing to H
        E.append(np.amax(np.abs(U_comp - U_true)))
        H.append(dr)

    # create log-log error plot
    utils.plot_error(E, H, problem=3)


if __name__ == '__main__':
    Nx = [101, 1001, 10001, 100001]
    start = time.time()
    main(Nx)
    print("Elapsed Time: ", time.time()-start)

