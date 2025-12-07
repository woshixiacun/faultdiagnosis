import numpy as np
from scipy.signal import correlate


def TT(y, fs):
    M = fs
    NA = correlate(y, y, mode='full')
    NA = NA[len(NA)//2:]
    for lag in range(1, len(NA)):
        if NA[lag] <= 0:
            zeroposi = lag
            break
    NA = NA[zeroposi:]
    T = zeroposi + np.argmax(NA)
    return T


def CK(x, T, M=2):
    x = np.array(x).flatten()
    N = len(x)
    x_shift = np.zeros((M+1, N))
    x_shift[0] = x

    for m in range(1, M+1):
        x_shift[m, T:] = x_shift[m-1, :-T]

    return np.sum(np.prod(x_shift, axis=0)**2) / (np.sum(x**2)**(M+1))


def max_IJ(X):
    idx = np.unravel_index(np.argmax(X), X.shape)
    return idx[0], idx[1], X[idx]
