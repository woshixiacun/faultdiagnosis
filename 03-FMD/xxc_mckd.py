# （MCKD核心）
import numpy as np
from scipy.signal import hilbert, lfilter
from numpy.linalg import inv
from utils import TT


def xxc_mckd(fs, x, f_init, termIter=30, T=None, M=3, plotMode=0):
    x = np.array(x).flatten()
    L = len(f_init)
    N = len(x)

    if T is None:
        env = np.abs(hilbert(x)) - np.mean(np.abs(hilbert(x)))
        T = int(TT(env, fs))

    XmT = np.zeros((L, N, M+1))
    for m in range(M+1):
        for l in range(L):
            if l == 0:
                XmT[l, m*T:N, m] = x[:N-m*T]
            else:
                XmT[l, 1:, m] = XmT[l-1, :-1, m]

    Xinv = inv(XmT[:, :, 0] @ XmT[:, :, 0].T)

    f = f_init.copy()
    y_final = []
    f_final = []
    ckIter = []

    for n in range(termIter):
        y = (f @ XmT[:, :, 0]).T
        f_final.append(f.copy())

        yt = np.zeros((N, M+1))
        yt[:, 0] = y
        for m in range(1, M+1):
            yt[T:, m] = yt[:-T, m-1]

        beta = np.prod(yt, axis=1)
        alpha = np.zeros((N, M+1))
        for m in range(M+1):
            idx = [i for i in range(M+1) if i != m]
            alpha[:, m] = (np.prod(yt[:, idx], axis=1)**2) * yt[:, m]

        Xalpha = sum(XmT[:, :, m] @ alpha[:, m] for m in range(M+1))
        f = (np.sum(y**2) / (2*np.sum(beta**2))) * Xinv @ Xalpha
        f /= np.linalg.norm(f)

        ck = np.sum(beta**2) / (np.sum(y**2)**(M+1))
        ckIter.append(ck)

        env = np.abs(hilbert(y)) - np.mean(np.abs(hilbert(y)))
        T = int(TT(env, fs))

        XmT = np.zeros((L, N, M+1))
        for m in range(M+1):
            for l in range(L):
                if l == 0:
                    XmT[l, m*T:N, m] = x[:N-m*T]
                else:
                    XmT[l, 1:, m] = XmT[l-1, :-1, m]

        Xinv = inv(XmT[:, :, 0] @ XmT[:, :, 0].T)
        y_final.append(lfilter(f_final[-1], 1, x))

    return np.array(y_final).T, np.array(f_final).T, ckIter, T
