import numpy as np

##############################
# Part 1
##############################
# Write your answers to the math problems below. To represent exponents, use the ^
#symbol.
# 1. Partial Derivative with Respect to x: 2y - z + y^2
#    Partial Derivative with Respect to y: 2x + 2xy
#    Partial Derivative with Respect to z: -x
#
# 2. Partial Derivative with Respect to u: sin(v)
#    Partial Derivative with Respect to v: ucos(v) + 2/u
#
# 3. Matrix B is a 7 x 3 matrix. (7 rows 3 columns)
#
# 4. Partial Derivative with Respect to x: -3
#    Partial Derivative with Respect to y: -1

##############################
# Part 2
##############################

def problem1 (A, B, C):
    return np.dot(A, B) - C

def problem2 (A):
    return np.ones((A.shape[0],))

def problem3 (A):
    result = A.copy()
    np.fill_diagonal(result, 0)
    return result

def problem4 (A, i):
    return np.sum(A[i])

def problem5 (A, c, d):
    mask = (A >= c) & (A <= d)
    return np.mean(A[mask])

def problem6 (A, k):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    indices = np.argsort(-np.abs(eigenvalues))[:k]
    return eigenvectors[:, indices]

def problem7 (A, x):
    return np.linalg.solve(A, x)

def problem8 (x, k):
    return np.tile(x, (k, 1)).T

def problem9 (A):
    return A[np.random.permutation(A.shape[0])]

def problem10 (A):
    return np.mean(A, axis=1)

def problem11 (n, k):
    A = np.random.randint(0, k + 1, size=n)
    A[A % 2 == 0] = -1
    return A

def problem12 (A, b):
    b = b[:, np.newaxis]
    return A + b

def problem13 (A):
    n, m, _ = A.shape
    return A.reshape(n, -1).T
