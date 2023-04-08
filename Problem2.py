'''
Author: William Powell
Description: Implements the conjugate gradient method to solve the Poisson
equation with homogeneous Dirichlet boundary conditions on the unit square in
R^2
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
import utils
import os


# returns sparse operator matrix A. Much faster than calling op_matrix_multiply
def op_matrix_sparse(Nx, dx):
    dxx =dx**2
    row = []
    col = []
    vals = []
    N = Nx-2
    directions = {(0, 0): -20/(6*dxx), (1, 0): 4/(6*dxx), (-1, 0): 4/(6*dxx),
                  (0, 1): 4/(6*dxx), (0, -1): 4/(6*dxx),
                  (1, 1): 1/(6*dxx), (1, -1): 1/(6*dxx),
                  (-1, -1): 1/(6*dxx), (-1, 1): 1/(6*dxx)}

    # order the entries row-wise by iterating through the x-direction first, then the y-direction
    for ix in range(N):
        for iy in range(N):
            ie0 = ix + iy * N

            # exclude out of range indices if on the boundary
            indices = [ix + d1 + (iy + d2) * N for d1, d2 in directions if 0 <= ix + d1 < N and 0 <= iy + d2 < N]

            row.extend([ie0 for i in range(len(indices))])
            col.extend(indices)
            vals.extend([directions[(d1, d2)] for d1, d2 in directions if 0 <= ix + d1 < N and 0 <= iy + d2 < N])

    A = sp.csr_matrix((vals, (row, col)), shape=(N**2, N**2))

    return A


# Returns the column vector F in the equation Au = F
def target_vector(func, X, Y, Nx):
    # we don't include separate rows for the boundaries so the vector has length (Nx-2)**2
    N = Nx - 2

    # the Nx x Nx discrete version of the right hand side f evaluated on the meshgrid X Y
    F_m = func(X, Y)

    # initialize empty target F
    F = np.zeros(N**2)

    # no need to account for boundary conditions here because they are zero.
    # take only the points of F evaluated at the interior of the domain
    for ix in range(1, Nx-1):
        for iy in range(1, Nx-1):
            ie0 = ix-1 + (iy-1)*N
            F[ie0] = F_m[ix, iy] + (1/12)*(F_m[ix, iy-1] + F_m[ix-1, iy]
                                           + F_m[ix, iy+1] + F_m[ix+1, iy]
                                           - 4*F_m[ix, iy])
    return F


# the right hand side of the Poison equation
def f(X, Y):
    return 5*np.pi**2*np.sin(np.pi*(X-1))*np.sin(2*np.pi*(Y-1))


# The scalar to be used in the CG descent
def search_scalar(r, p, w):
    a = np.dot(r, r)
    b = np.dot(p, w)
    return a/b, a


# implements conjugate gradient algorithm
def CG(u0, A, F, N, dx, tolerance=0.01, max_iter=100, record_res_error=False):
    r = F + A.dot(u0)
    p = r
    u = u0
    res = []
    count = 0
    while count < max_iter:
        w = -A.dot(p)
        alpha, prev_res_IP = search_scalar(r, p, w)
        u = u + alpha*p
        r = r - alpha*w
        curr_res_IP = np.dot(r, r)
        if record_res_error:
            res.append([curr_res_IP**(1/2), count])
        if curr_res_IP**(1/2) < tolerance:
            break
        b = curr_res_IP/prev_res_IP
        p = r + b*p
        count += 1
        #print("iteration count: {}/{}".format(count, N**2))

    print('CG complete for step size {:.2} at iteration {}'.format(dx, count+1))

    if record_res_error:
        return u, res
    else:
        return u


def vector_to_matrix(U, Nx):
    N = Nx-2
    u = np.zeros((Nx, Nx))
    for ix in range(1, Nx-1):
        for iy in range(1, Nx-1):
            ie = ix-1 + (iy-1)*N
            u[ix, iy] = U[ie]

    return u


def plot2D(u, X, Y):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, u, linewidth=0)
    plt.show()
    plt.close()


def plot_res_error(res_error, dx):
    res_error = np.array(res_error)

    plt.figure()

    ax = plt.subplot()

    ax.plot(res_error[:,1], res_error[:, 0])

    ax.set_xlabel(r'Iteration $k$')
    ax.set_ylabel(r'Residual $L_2$ norm')

    directory = os.path.join(os.getcwd(), 'pic')
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, 'problem2_res_error.png')

    plt.savefig(path)

    plt.close()

    #plt.show()


def run(grid_sizes, res_plot_size):
    # create lists to store error and step size
    E = []
    H = []

    for Nx in grid_sizes:
        # create grid in one dimension
        x = np.linspace(0, 1, Nx)
        dx = x[1] - x[0]

        # create the mesh
        X, Y = np.meshgrid(x, x, sparse=True)

        # operator matrix dimensions
        N = Nx-2

        # initial guess in vector form. Must appropriate dimensions since the matrix A is (Nx-2)^2 X (Nx-2)^2
        u0 = np.zeros(N**2)

        # get vector and matrix in equation AU = F
        F = target_vector(f, X, Y, Nx)
        A = op_matrix_sparse(Nx, dx)

        # compute solution using conjugate gradient algorithm.
        # Our method is O(h^4) so we set the stopping tolerance to h^4.
        # Set the maximum iterations to the dimension of A.
        # If Nx==res_plot_size plot the L_2 residual error as a function of iteration
        start = time.time()
        if Nx == res_plot_size:
            U, res_error = CG(u0, A, F, Nx, dx, tolerance=dx**4, max_iter=N**2, record_res_error=True)
            print("Elapsed time: {:.2} for step size {:e}".format(time.time() - start, dx))
            plot_res_error(res_error, dx)
        else:
            U = CG(u0, A, F, Nx, dx, tolerance=dx**4, max_iter=Nx**2)
            print("Elapsed time: {:.2} for step size {:e}".format(time.time() - start, dx))

        # computed solution on X, Y mesh
        u_comp = vector_to_matrix(U, Nx)

        # true solution
        u_true = (1/(5*np.pi**2))*f(X, Y)
        #plot2D(u_comp, X, Y)

        E.append(np.amax(u_comp - u_true))
        H.append(dx)

    # create the log-log error plot
    utils.plot_error(E, H, problem=2)


if __name__ == "__main__":
    run([11, 101, 1001], 1001)