from numpy import *

def lamb(u, N):
    if(u == 0):
        return sqrt(1/N)
    else:
        return sqrt(2/N)

def dct1(A, i, j):
    F = 0.0
    N = A.shape[0]

    for x in range(0, N):
        for y in range(0, N):
            F +=            cos( 
                                ((math.pi * i) * (2*x+1)) /
                                (2*N)
                                ) * \
                            cos( 
                                ((math.pi * j) * (2*y+1)) / 
                                (2*N)
                                ) * \
                            A[x, y]
    return F

def dct2(A):
    N = A.shape[0]
    coeffs = empty([N,N])

    for i in range(0, N):
        for j in range(0, N):
            coeffs[i, j] = lamb(i, N) * lamb (j, N) * dct1(A, i, j)
                            
    return coeffs

def idct2(A):
    N = A.shape[0]
    coeffs = uint8(empty([N,N]))

    for i in range(0, N):
        for j in range(0, N):
            coeffs[i, j] = round(idct1(A, i, j))
                            
    return coeffs

def idct1(A, i, j):
    F = 0.0
    N = A.shape[0]

    for x in range(0, N):
        for y in range(0, N):
            F +=            lamb(x, N) * lamb (y, N) * \
                            cos( 
                                ((math.pi * i) * (2*x+1)) /
                                (2*N)
                                ) * \
                            cos( 
                                ((math.pi * j) * (2*y+1)) / 
                                (2*N)
                                ) * \
                            A[x, y]
    return F

def cosines(N):
    cosines = empty([N, N])
    for x in range(0, N):
        for y in range(0, N):
            cosines[x, y] = lamb(x, N) * \
                            cos( 
                                ((math.pi * x) * (2*y+1)) /
                                (2*N)
                                ) 
    return cosines