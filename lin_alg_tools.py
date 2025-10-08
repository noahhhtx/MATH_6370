# Noah Harrison 2046687
# this script contains several common linear algebra functions

def print_matrix(A_, precision=20, newline=True):
    for i in range(len(A_)):
        for j in range(len(A_[0])):
            print(f"{A_[i][j]:.{precision}f}", end=" ")
        print()
    if newline:
        print()

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

def init_matrix(M, N = None):

    if N is None:
        N = M
    mat = []
    for i in range(M):
        temp = []
        for k in range(N):
            temp.append(0)
        mat.append(temp)
    return mat

def init_identity_matrix(M):

    mat = init_matrix(M)
    for i in range(M):
        mat[i][i] = 1
    return mat