import numpy as np
from scipy.stats import gmean
cimport cython
from libc.math cimport exp, log

def swgm_cython_wrapper(X, beta=0.3, max_gamma=20.0):
    return swgm(X, beta, max_gamma)

@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef swgm(double[:,:,::1] X, double beta, double max_gamma):
    cdef:
        Py_ssize_t P = X.shape[0] # Eixo dos espectrogramas
        Py_ssize_t K = X.shape[1] # Eixo das frequências
        Py_ssize_t M = X.shape[2] # Eixo do tempo

        Py_ssize_t p, k, m, aux_p
        double epsilon = 1e-10

    print("shape", P, K, M)

    # Tensor pré-calculado de logaritmos.
    log_X_ndarray = np.log(np.asarray(X) + epsilon, dtype=np.double)
    cdef double[:, :, :] log_X = log_X_ndarray
    
    # Matriz pré calculada de soma de logaritmos.
    sum_log_X_ndarray = np.sum(log_X_ndarray, axis=0) / (P - 1)
    cdef double[:, :] sum_log_X = sum_log_X_ndarray

    # Tensor de pesos.
    gammas_ndarray = np.zeros((P, K, M), dtype=np.double)
    cdef double[:,:,:] gammas = gammas_ndarray
    
    # Cálculo dos pesos
    for k in range(K):
        for m in range(M):
            for p in range(P):
                gammas[p, k, m] = sum_log_X[k, m] - log_X[p, k, m] * P / (P - 1)
                gammas[p, k, m] = exp(gammas[p, k, m] * beta)
                if gammas[p, k, m] > max_gamma:
                    gammas[p, k, m] = max_gamma

    # Cálculo das médias geométricas ponderadas.
    return gmean(X, axis=0, weights=gammas_ndarray)