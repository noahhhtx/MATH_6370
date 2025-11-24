# Noah Harrison 2046687
# This script will compute the Thomas algorithm to solve a linear system Ax = b

import copy
import lin_alg_tools as lat

'''
I'm sure you're wondering why I wrote two factorization functions. I wrote the two-matrix factorization function first 
since I found it more intuitive, and I wrote the one-matrix factorization function later since that's how the result of 
a Thomas factorization is generally stored.
'''

def factorization_onematrix(A):

    # this function returns only a single matrix: A_, which contains the LU decomposition

    # create a deep copy of A to preserve the original matrix
    A_ = copy.deepcopy(A)

    # now we can make two assumptions:
    # 1) we don't need to change a(1) since alpha(1) = a(1)
    # 2) the c values also remain unchanged.

    for i in range(1, len(A)):

        # get relevant alpha and b values to compute beta
        alpha_prev = A_[i-1][i-1]
        b = A_[i][i-1]

        # beta(i) = b(i) / alpha(i-1)
        beta = b / alpha_prev

        # get relevant a and c values to compute alpha
        a = A_[i][i]
        c = A_[i-1][i]

        # alpha(i) = a(i) - ( beta(i) * c(i-1) )
        alpha = a - (beta * c)

        # assign alpha and beta values
        A_[i][i-1] = beta
        A_[i][i] = alpha

    return A_

def factorization_twomatrices(A):

    # this function returns two matrices: L & U

    L = lat.init_identity_matrix(len(A))
    U = lat.init_matrix(len(A))

    # set c values in U to c values in A
    for i in range(len(A)-1):
        U[i][i+1] = A[i][i+1]

    # alpha_1 = a1
    U[0][0] = A[0][0]

    # compute alpha and beta values
    for i in range(1, len(A)):

        # get relevant alpha and b values to compute beta
        alpha_prev = U[i-1][i-1]
        b_val = A[i][i-1]

        # beta(i) = b(i) / alpha(i-1)
        beta = b_val / alpha_prev

        # get relevant a and c values to compute alpha
        a_val = A[i][i]
        c_val = U[i-1][i]

        # alpha(i) = a(i) - ( beta(i) * c(i-1) )
        alpha = a_val - (beta * c_val)

        # assign alpha and beta values
        L[i][i-1] = beta
        U[i][i] = alpha

    return L, U

def solve(L, U, b):

    # this function takes decomposed L & U matrices, but you can also just pass A twice.

    y = lat.init_matrix(len(b), 1)

    # y(1) = b(1)
    y[0][0] = b[0][0]

    for i in range(1, len(b)):
        # y(i) = f(i) - ( beta(i) * y(i-1) )
        y[i][0] = b[i][0] - (L[i][i-1] * y[i-1][0])

    x = lat.init_matrix(len(y), 1)
    n = len(x) - 1

    # x(n) = y(n) / alpha(n)
    x[n][0] = y[n][0] / U[n][n]

    for i in range(n-1, -1, -1):
        # x(i) = ( y(i) - ( c(i) * x(i+1) ) ) / alpha(i)
        x[i][0] = ( y[i][0] - ( U[i][i+1] * x[i+1][0] ) ) / U[i][i]

    return x

if __name__ == '__main__':

    mat = [

        [2, 1, 0, 0],
        [2, -4, 0, 0],
        [0, -1, -3, 1],
        [0, 0, -1, 2]

    ]

    b = [

        [3],
        [-2],
        [-3],
        [1]

    ]

    print("Original Matrix:")
    lat.print_matrix(mat,0)

    # test one-matrix function
    print("Thomas Factorization with one-matrix function")
    A = factorization_onematrix(mat)
    lat.print_matrix(A, 5)

    # test two-matrix function
    print("Thomas Factorization with two-matrix function")
    L, U = factorization_twomatrices(mat)
    print("L:")
    lat.print_matrix(L, 5)
    print("U:")
    lat.print_matrix(U, 5)

    # solve Ax=b
    print("(b) Vector:")
    lat.print_matrix(b, 0)
    print("Solution (x) for Ax=b:")
    x = solve(A, A, b)
    lat.print_matrix(x, 20)
    print("which becomes")
    lat.print_matrix(x, 0, False)
    print("when rounded.\n")
    # this should return b
    print("Result of multiplying A by x:")
    lat.print_matrix(lat.matrix_mult(mat, x), 20)
