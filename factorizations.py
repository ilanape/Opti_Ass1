import numpy as np
import scipy.linalg

# Question 1b
A = np.array([[1, 2, 3, 4], [2, 4, -4, 8], [-5, 4, 1, 5], [5, 0, -3, -7]])

ATA = np.transpose(A) @ A
# eigenvectors and eigenvalues of ATA
eigvals, eigvecs = scipy.linalg.eig(ATA)
maxEigenVal = max(eigvals)
# the index of the largest eigenvalue in eigenvals
# the columns in eigenvecs corresponds to eigenvals
index = np.where(eigvals == maxEigenVal)
x = eigvecs[:, index[0]]
print(x)

# Question 4.a
A = np.array([[2, 1, 2], [1, -2, 1], [1, 2, 3], [1, 1, 1]])
b = [6, 1, 5, 2]

AT = np.transpose(A)
L = np.linalg.cholesky(AT @ A)
# A^TA=LL^T from the factorization
# denote y = L^Tx
# L in lower triangular - solve by forward substitution
y = scipy.linalg.solve_triangular(L, AT @ b, lower=True)
# L^T in upper triangular - solve by back substitution
x = scipy.linalg.solve_triangular(np.transpose(L), y, lower=False)
print(x)

# Question 4.b
# QR factorization
Q, R = np.linalg.qr(A)
QTb = np.transpose(Q) @ b
# Rx=QTb normal equation
# R is upper triangular - solve by back substitution
x = scipy.linalg.solve_triangular(R, QTb, lower=False)
print(x)

# SVD factorization
U, S, VT = np.linalg.svd(A, full_matrices=False)
UTb = np.transpose(U) @ b
# SVTx=UTb the normal equation
# denote y=VTx
# Sy=UTb is a diagonal system
# y = (1/S)UTb trivial inversion of a diagonal matrix
y = np.diag(1 / S) @ UTb
# x = VT
x = np.transpose(VT) @ y
print(x)

# Question 4.c
r = A @ x - b
print(r)
print(np.transpose(A) @ r)

# Question 4.d
W = np.eye(4, dtype=int)
AT = np.transpose(A)
ATWA = AT @ W @ A
ATWb = AT @ W @ b
# ATWAx=ATWb the normal equation
x = np.linalg.solve(ATWA, ATWb)
r1 = (A @ x - b)[0]
while np.absolute(r1) >= pow(10, -3):
    W[0, 0] += 1
    # re-calculate
    ATWA = AT @ W @ A
    ATWb = AT @ W @ b
    x = np.linalg.solve(ATWA, ATWb)
    r1 = (A @ x - b)[0]

print(W)
print(r1)
print(x)


# Question 5.a
# QR factorization using GS
def qr_grams(a):
    n = len(a[0])
    # initialize
    R = np.zeros((n, n))
    a0 = a[:, 0]
    R[0, 0] = np.linalg.norm(a0)
    q0 = a0 / R[0, 0]
    Q = np.array([q0]).T
    for i in range(1, n):
        ai = a[:, i]
        qi = ai
        for j in range(0, i):
            qj = Q[:, j]
            R[j, i] = np.transpose(qj) @ ai
            qi = qi - (R[j, i] * qj)
        R[i, i] = np.linalg.norm(qi)
        qi = qi / R[i, i]
        Q = np.column_stack((Q, qi))
    return Q, R


# QR factorization using MGS
def qr_mod_grams(a):
    n = len(a[0])
    # initialize
    R = np.zeros((n, n))
    a0 = a[:, 0]
    R[0, 0] = np.linalg.norm(a0)
    q0 = a0 / R[0, 0]
    Q = np.array([q0]).T
    for i in range(1, n):
        ai = a[:, i]
        qi = ai
        for j in range(0, i):
            qj = Q[:, j]
            # difference from regular gs in dependence on changed qi
            R[j, i] = np.transpose(qj) @ qi
            qi = qi - (R[j, i] * qj)
        R[i, i] = np.linalg.norm(qi)
        qi = qi / R[i, i]
        Q = np.column_stack((Q, qi))
    return Q, R


# Question 5.b
e = 1
# e = pow(10, -10)
A = np.array([[1, 1, 1], [e, 0, 0], [0, e, 0], [0, 0, e]])
Q1, R1 = qr_grams(A)
print(np.linalg.norm(np.transpose(Q1) @ Q1 - np.eye(3, 3), 'fro'))
Q2, R2 = qr_mod_grams(A)
print(np.linalg.norm(np.transpose(Q2) @ Q2 - np.eye(3, 3), 'fro'))
