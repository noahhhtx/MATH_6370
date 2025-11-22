# Noah Harrison 2046687
# this script contains several common linear algebra functions

import copy
import math

def print_matrix(A_, precision=20, newline=True):

    for i in range(len(A_)):
        for j in range(len(A_[0])):
            print(f"{A_[i][j]:.{precision}f}", end=" ")
        print()
    if newline:
        print()

def matrix_add(A,B):

    '''
    A: matrix of size MxN
    B: matrix of size MxN
    returns a matrix of size MxN
    '''

    if len(A) == 0 or len(B) == 0:
        raise Exception("One of the matrices is empty")

    if len(A) != len(B):
        raise Exception("Row dimensions are not equivalent.")

    if len(A[0]) != len(B[0]):
        raise Exception("Column dimensions are not equivalent.")

    new_mat = init_matrix(len(A), len(A[0]))

    for i in range(len(A)):
        for j in range(len(A[0])):
            new_mat[i][j] = A[i][j] + B[i][j]

    return new_mat

def matrix_mult(A,B):

    '''
    A: matrix of size MxP
    B: matrix of size PxN
    returns a matrix of size MxN
    '''

    if len(A) == 0 or len(B) == 0:
        raise Exception("One of the matrices is empty.")

    if len(A[0]) != len(B):
        raise Exception(f"Dimensions of matrices ({len(A)} x {len(A[0])} and {len(B)} x {len(B[0])}) do not enable multiplication.")

    P = len(A[0])
    M = len(A)
    N = len(B[0])

    result = []

    for m in range(M):
        temp = []
        for n in range(N):
            sum = 0
            for i in range(P):
                sum += (A[m][i] * B[i][n])
            temp.append(sum)
        result.append(temp)

    return result

def scalar_mult(A, a):

    if len(A) == 0:
        raise Exception("Matrix is empty.")

    A_ = init_matrix(len(A), len(A[0]))
    for i in range(len(A_)):
        for j in range(len(A_[0])):
            A_[i][j] = a * (A[i][j])

    return A_


def transpose(A):

    '''
    A: matrix of size MxN
    returns a matrix of size NxM
    '''

    result = []
    for i in range(len(A[0])):
        temp = []
        for j in range(len(A)):
            temp.append(A[j][i])
        result.append(temp)

    return result

def euclid_norm(A):

    if len(A) == 0:
        raise Exception("Matrix is empty")

    res = 0

    if len(A) >= 1 and len(A[0]) == 1:
        res = matrix_mult(transpose(A), A)

    elif len(A) == 1 and len(A[0]) >= 1:
        res = matrix_mult(A, transpose(A))

    else:
        raise Exception("Improper matrix dimensions.")

    return math.sqrt(res[0][0])

def init_matrix(M, N = None, fill_val = 0):

    if N is None:
        N = M
    mat = []
    for i in range(M):
        temp = []
        for k in range(N):
            temp.append(fill_val)
        mat.append(temp)
    return mat

def init_identity_matrix(M):

    mat = init_matrix(M)
    for i in range(M):
        mat[i][i] = 1
    return mat

def hilbert_matrix(M):

    mat = init_matrix(M)

    for i in range(M):
        for j in range(M):
            mat[i][j] = 1 / ( i + j + 1 )

    return  mat

def diag(A):

    if len(A) == 0:
        raise Exception("Matrix is empty.")

    if len(A) != len(A[0]):
        raise Exception("Matrix is not square")

    mat = init_matrix(len(A))

    for i in range(len(A)):
        mat[i][i] = A[i][i]

    return mat

def residual_linear(A, x, b):

    '''
    A: Matrix of size MxM
    x: Vector of size M
    b: Vector of size M
    '''

    return matrix_add(b, scalar_mult(matrix_mult(A, x), -1))

def residual_eigenvalue(A, q, v):

    a = matrix_mult(A, q)
    b = scalar_mult(q, v)

    return matrix_add(a, scalar_mult(b, -1))

def trace(A):

    s = 0

    if len(A) == 0:
        return s

    if len(A) != len(A[0]):
        raise Exception("Matrix is not square.")

    for i in range(len(A)):
        s += A[i][i]

    return s

def inverse(A):

    A_ = copy.deepcopy(A)

    if len(A) == 0:
        raise Exception("Matrix is empty.")

    if len(A) != len(A[0]):
        raise Exception("Matrix is not square.")

    B = init_identity_matrix(len(A))
    alpha = 0

    for i in range(len(A)):

        k = i + 1
        alpha = trace( matrix_mult(A_, B) ) / k

        if i < len(A) - 1:
            # do not update B the last time.
            B = matrix_add( scalar_mult( matrix_mult(A_, B), -1 ), scalar_mult( init_identity_matrix(len(A)), alpha ) )

    if alpha == 0:
        raise Exception("Matrix is not invertible.")

    return scalar_mult(B, 1 / alpha)

def solve_pivot(A, b):

    if len(A) == 0:
        raise Exception("Matrix is empty.")

    if len(A) != len(A[0]):
        raise Exception("Matrix has improper dimensions.")

    A_ = copy.deepcopy(A)
    b_ = copy.deepcopy(b)

    # 1: compute upper triangular matrix

    for k in range(len(A)):

        max_val = A_[k][k]
        best_ind = k

        for i in range(k+1, len(A)):

            if abs(A_[i][k]) > abs(max_val):
                max_val = A_[i][k]
                best_ind = i

        if best_ind != k:

            temp_b = b_[best_ind][0]
            b_[best_ind][0] = b_[k][0]
            b_[k][0] = temp_b

            for j in range(len(A)):
                temp_a = A_[best_ind][j]
                A_[best_ind][j] = A_[k][j]
                A_[k][j] = temp_a

        for i in range(k+1, len(A)):

            m = A_[i][k] / A_[k][k]

            for j in range(0, len(A)):
                A_[i][j] -= ( m * A_[k][j] )

            b_[i][0] -= ( m * b_[k][0] )

    # 2: solve using back-substitution

    x = init_matrix(len(A), 1)
    x[len(A)-1][0] = b_[len(A)-1][0] / A_[len(A)-1][len(A)-1]

    for i in range(len(A)-2, -1, -1):

        sum = 0

        for j in range(i+1, len(A)):
            sum += ( A_[i][j] * x[j][0] )

        x[i][0] = ( 1 / A_[i][i] ) * ( b_[i][0] - sum )

    return x