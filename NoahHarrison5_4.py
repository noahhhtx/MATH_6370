# Noah Harrison 2046687
# This script will use the power method to find the largest eigenvalue of a matrix A

import copy
import math
import lin_alg_tools as lat

threshold = 10**(-10)

def unit_vector(M):

    vec = lat.init_matrix(M, 1)
    vec[0][0] = 1

    return vec

def rayleigh(A, q):

    return lat.matrix_mult( lat.transpose(q), lat.matrix_mult( A, q ) )[0][0]

def power_method(A, q_init, t = 10**(-8), maxiter=10000):

    '''
    A: Matrix of size MxM
    q_init: Vector of size M
    t: Scalar, threshold value
    '''

    if len(A) == 0:
        raise Exception("Matrix is empty.")

    if len(A) != len(A[0]):
        raise Exception("Matrix has improper dimensions.")

    q_prev = copy.deepcopy(q_init)
    res = 99999
    v = None

    n = 1

    while res >= t and n <= maxiter:

        z = lat.matrix_mult(A, q_prev)
        q = lat.scalar_mult(z, 1/lat.euclid_norm(z))
        v = rayleigh(A, q)

        print(f"Iteration {n}: {v}")
        res = lat.euclid_norm(lat.residual_eigenvalue(A, q, v))

        q_prev = q

        n += 1

    if n > maxiter:
        print("Method did not converge.")

    print()

    return v, q_prev

if __name__ == '__main__':

    matrices = [

        [
            [1,2,0],
            [1,0,0],
            [0,1,0]
        ],
        [
            [0.1,3.8,0],
            [1,0,0],
            [0,1,0]
        ],
        [
            [0,-1,0],
            [1,0,0],
            [0,1,0]
        ]

    ]

    q_0 = lat.scalar_mult(lat.transpose([[1,2,1]]), 1/math.sqrt(6))

    for matrix in matrices:

        print("Matrix")
        lat.print_matrix(matrix, 1)
        eigenvalue, eigenvector = power_method(matrix, q_0, threshold)
        print("Largest eigenvalue:", eigenvalue)
        print("Corresponding eigenvector:")
        lat.print_matrix(eigenvector)
    