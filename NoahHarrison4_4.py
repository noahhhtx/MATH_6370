# Noah Harrison 2046687
# This script will use the inverse power method to find the extreme eigenvalues of a matrix A (and subsequently
# compare them to NumPy's eigenvalues) and use the preconditioned gradient method to solve Ax=b.

import numpy as np
import lin_alg_tools as lat

threshold = 10**(-8)
verb = False

def unit_vector(M):

    vec = lat.init_matrix(M, 1)
    vec[0][0] = 1

    return vec

def rayleigh(A, q):

    return lat.matrix_mult( lat.transpose(q), lat.matrix_mult( A, q ) )[0][0]

def power_method(A, mu = 0, t = 10**(-8), verbose = True):

    '''
    A: Matrix of size MxM
    mu: Scalar, used to compute A - (mu)I
    t: Scalar, threshold value
    '''

    if len(A) == 0:
        raise Exception("Matrix is empty.")

    if len(A) != len(A[0]):
        raise Exception("Matrix has improper dimensions.")

    new_mat = lat.matrix_add( A, lat.scalar_mult(lat.init_identity_matrix(len(A)), -mu) )
    q_prev = unit_vector(len(A))
    res = 99999
    v = None

    n = 1

    while res > t:

        z = lat.solve_pivot(new_mat, q_prev)
        q = lat.scalar_mult(z, 1/lat.euclid_norm(z))
        v = rayleigh(A, q)

        if verbose:
            print(f"Iteration {n}: {v}")
        res = lat.euclid_norm(lat.residual_eigenvalue(A, q, v))

        q_prev = q

        n += 1

    if verbose:
        print()

    return v, q_prev

def gradient(A, b, P, t = 10**(-8)):

    '''
    A: Matrix of size MxM
    b: Vector of size M
    P: Matrix of size MxM; preconditioner
    t: Scalar, threshold value
    '''

    if len(A) == 0:
        raise Exception("Matrix is empty.")

    if len(A) != len(A[0]):
        raise Exception("Matrix has improper dimensions.")

    x_0 = lat.init_matrix(len(A), 1, 0)
    r_0 = lat.residual_linear(A, x_0, b)
    b_norm = lat.euclid_norm(b)

    x_prev = x_0
    r_prev = r_0

    n = 1
    norm_ratio = 9999

    while norm_ratio > t:

        z = lat.solve_pivot(P, r_prev)
        alpha = ( lat.matrix_mult( lat.transpose(z), r_prev )[0][0] ) / ( rayleigh(A, z) )
        x = lat.matrix_add( x_prev, lat.scalar_mult(z, alpha) )
        r = lat.matrix_add( r_prev, lat.scalar_mult( lat.matrix_mult( A, z ) , -alpha ) )

        r_norm = lat.euclid_norm(r)
        norm_ratio = r_norm / b_norm

        x_prev = x
        r_prev = r

        n += 1

    print(f"Convergence reached in {n} iterations: {norm_ratio}")
    print()

    return x_prev

if __name__ == '__main__':

    A = lat.hilbert_matrix(5)
    lat.print_matrix(A, 10)

    eigenvalues_numpy, eigenvectors_numpy = np.linalg.eig(A)

    # Computing eigenvalues and comparing them to NumPy's results.
    eigenvalue_small, eigenvector_small = power_method(A, 0, threshold, True)
    print(f"Smallest eigenvalue of Hilbert(5) according to algorithm: {eigenvalue_small}\n")
    if verb:
        print(f"Corresponding eigenvector:")
        lat.print_matrix(eigenvector_small)
        lat.print_matrix( lat.matrix_mult(A, eigenvector_small) )
        lat.print_matrix( lat.scalar_mult(eigenvector_small, eigenvalue_small) )
    print(f"Smallest eigenvalue of Hilbert(5) according to NumPy: {min(eigenvalues_numpy)}\n")
    if verb:
        print(f"Corresponding eigenvector:")
        small_idx = np.where(eigenvalues_numpy==min(eigenvalues_numpy))[0]
        np_vector_small = eigenvectors_numpy[:, small_idx]
        lat.print_matrix(np_vector_small)
        lat.print_matrix( lat.matrix_mult(A, np_vector_small) )
        lat.print_matrix( lat.scalar_mult(np_vector_small, min(eigenvalues_numpy)) )
    eigenvalue_large, eigenvector_large = power_method(A, 2, threshold, True)
    print(f"Largest eigenvalue of Hilbert(5) according to algorithm: {eigenvalue_large}\n")
    if verb:
        print(f"Corresponding eigenvector:")
        lat.print_matrix(eigenvector_large)
        lat.print_matrix( lat.matrix_mult(A, eigenvector_large) )
        lat.print_matrix( lat.scalar_mult(eigenvector_large, eigenvalue_large) )
    print(f"Largest eigenvalue of Hilbert(5) according to NumPy: {max(eigenvalues_numpy)}\n")
    if verb:
        print(f"Corresponding eigenvector:")
        large_idx = np.where(eigenvalues_numpy==max(eigenvalues_numpy))[0]
        np_vector_large = eigenvectors_numpy[:, large_idx]
        lat.print_matrix(np_vector_large)
        lat.print_matrix( lat.matrix_mult(A, np_vector_large) )
        lat.print_matrix( lat.scalar_mult(np_vector_large, max(eigenvalues_numpy)) )

    print(f"Eigenvalue ratio of A: {eigenvalue_large / eigenvalue_small}\n")
    # The eigenvalue ratio is large, which suggests that the gradient method will not converge quickly.

    b = lat.init_matrix(5, 1, 5)

    nonpreconditioned = lat.init_identity_matrix(5) # non-preconditioned method uses identity matrix.
    preconditioned = lat.diag(A) # we will use diag(A) as a preconditioner.

    p_inverse = np.linalg.inv(preconditioned)
    print("Inverse of Matrix according to NumPy:")
    lat.print_matrix(p_inverse)
    p_inverse_lat = lat.inverse(preconditioned)
    print("Inverse of Matrix according to my algorithm:")
    lat.print_matrix(p_inverse_lat)
    p_1_A = lat.matrix_mult(p_inverse_lat, A)
    p_1_A_np = lat.matrix_mult(p_inverse, A)

    precon_eigenval_numpy, precon_eigenvec_numpy = np.linalg.eig(p_1_A_np)

    precon_eigenvalue_small, precon_vector_small = power_method(p_1_A, 0, threshold, True)
    print(f"Smallest eigenvalue of P^(-1)A according to algorithm: {precon_eigenvalue_small}\n")
    print(f"Smallest eigenvalue of P^(-1)A according to NumPy: {min(precon_eigenval_numpy)}\n")

    precon_eigenvalue_large, precon_vector_large = power_method(p_1_A, 5, threshold, True)
    print(f"Largest eigenvalue of P^(-1)A according to algorithm: {precon_eigenvalue_large}\n")
    print(f"Largest eigenvalue of P^(-1)A according to NumPy: {max(precon_eigenval_numpy)}\n")

    print(f"Eigenvalue ratio of P^(-1)A: {precon_eigenvalue_large / precon_eigenvalue_small}\n")
    # This eigenvalue ratio is... smaller, so it may converge more quickly.

    print("\nAt this point, I am computing the results of the gradient methods (preconditioned and nonpreconditioned). Unless you want to see the results, you may terminate the program.\n")

    x_np = gradient(A, b, nonpreconditioned)
    lat.print_matrix(x_np, 5)
    lat.print_matrix( lat.matrix_mult(A, x_np) )
    x_p = gradient(A, b, preconditioned)
    lat.print_matrix(x_p, 5)
    lat.print_matrix( lat.matrix_mult(A, x_p) )

