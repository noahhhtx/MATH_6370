# Noah Harrison 2046687
# This script will compute the Jacobi method to solve a linear system Ax=b.

import math
import lin_alg_tools as lat

threshold = 10**(-8)
precision = 10

def jacobi(A, x, b, t, c = 5):

    '''
    A: Matrix of size MxM
    x: Vector of size M
    b: Vector of size M
    t: Integer or float; threshold values
    '''

    x_0 = x
    r_0 = lat.residual_linear(A, x_0, b)
    r_0_norm = lat.euclid_norm(r_0)

    n = 1
    norm_ratio = 99999

    x_prev = x_0

    while norm_ratio > t:

        x = lat.init_matrix(len(A), 1, 0)

        for i in range(len(A)):
            outer_val = 1 / A[i][i]
            inner_val = b[i][0]
            for j in range(len(A)):
                if j == i:
                    continue
                inner_val -= (A[i][j] * x_prev[j][0])
            x[i][0] = outer_val * inner_val

        r = lat.residual_linear(A, x, b)
        r_norm = lat.euclid_norm(r)
        norm_ratio = r_norm / r_0_norm

        print(f'Iteration {n}: {norm_ratio}')

        x_prev = x

        n += 1

    print()
    lat.print_matrix(x_prev, c)
    return x_prev

if __name__ == '__main__':

    A = [

        [3, 0, 1],
        [1, -4, 1],
        [-2, 2, 6]

    ]

    b = [

        [3],
        [1],
        [-1]

    ]

    x_0 = [

        [1],
        [1],
        [1]

    ]

    solution = jacobi(A, x_0, b, threshold, precision)
    lat.print_matrix(lat.matrix_mult(A, solution), precision)