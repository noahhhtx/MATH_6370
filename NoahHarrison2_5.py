# Noah Harrison 2046687
# This script will compute the Cholesky factorization of a symmetric & positive definite matrix.

import math
import lin_alg_tools as lat

def factorize(A):

    # we are allowed to assume that the matrix is spd.

    HT = lat.init_matrix(len(A))

    # h11 = sqrt(a11)
    HT[0][0] = math.sqrt(A[0][0])
    for i in range(1, len(A)):
        for j in range(i):
            # hij = ( aij - sum[k=1, j-1]( hik * hjk ) ) / hjj
            sum_ = 0
            for k in range(j):
                sum_ += (HT[i][k] * HT[j][k])
            HT[i][j] = ( A[i][j] - sum_ ) / HT[j][j]
        sum_ = 0
        # hii = sqrt( aii - sum[k=1, i-1]( (hik)^2 ) )
        for k in range(i):
            sum_ += ( (HT[i][k])**2 )
        difference = A[i][i] - sum_
        HT[i][i] = math.sqrt(difference)

    return HT

mat = [

    [4,12,-16],
    [12,37,-43],
    [-16,-43,98]

]

print("Original Matrix:")
lat.print_matrix(mat,0)

# Cholesky Factorization
H_t = factorize(mat)
H = lat.transpose(H_t)
print("Choleksy Factorization Results")
print("H^T:")
lat.print_matrix(H_t, 5)
print("H:")
lat.print_matrix(H, 5)
print("(H^T)H:")
lat.print_matrix(lat.matrix_mult(H_t, H), 5, False)
print("which becomes")
lat.print_matrix(lat.matrix_mult(H_t, H), 0, False)
print("when rounded.")
