import numpy as np
from scipy.signal import firwin, hilbert, windows, lfilter
from scipy.linalg import inv
from scipy.stats import zscore

def FMD(fs, x, FilterSize, CutNum, ModeNum, MaxIterNum):
    # Initialization
    freq_bound = np.arange(0, 1, 1/CutNum)
    temp_filters = np.zeros((FilterSize, CutNum))
    
    
    for n in range(len(freq_bound)):
        # 逻辑上按下面注释的（13、14）这两行做，但是这里版本有问题，所以忽略，直接看17行
        # w = windows.hann(FilterSize)
        # temp_filters[:, n] = firwin(FilterSize, [freq_bound[n]+1e-12, freq_bound[n]+1/CutNum-1e-12], window=w, pass_zero=False)
        
        # 每次初始化一列
        temp_filters[:, n] = firwin(FilterSize, [freq_bound[n]+1e-12, freq_bound[n]+1/CutNum-1e-12], window='hann', pass_zero=False)
        
    
    # Result initialization
    result = [{'IterCount': None, 'Iterations': [], 'CorrMatrix': None, 'ComparedModeNum': None, 'StopNum': None}]
    
    # Iterations
    temp_sig = np.tile(x[:, np.newaxis], (1, CutNum))
    itercount = 2
    
    while True:
        iternum = 2
        if itercount == 2:
            iternum = MaxIterNum - (CutNum - ModeNum) * iternum
        
        iter_result = {'IterCount': iternum, 'Iterations': [], 'CorrMatrix': None, 'ComparedModeNum': None, 'StopNum': None}
        
        for n in range(temp_filters.shape[1]):
            f_init = temp_filters[:, n]
            y_Iter, f_Iter, k_Iter, T_Iter = xxc_mckd(fs, temp_sig[:, n], f_init, iternum)
            
            iter_result['Iterations'].append({
                'y': y_Iter[:, -1],
                'f': f_Iter[:, -1],
                'k': k_Iter[:, -1],
                'fft': np.abs(np.fft.fft(f_Iter, axis=0))[:FilterSize//2, :],
                'peak_freq': np.argmax(np.abs(np.fft.fft(f_Iter, axis=0))[:FilterSize//2, :], axis=0)*(fs/FilterSize),
                'T': T_Iter
            })
        
        # Update temp_sig and temp_filters
        for n in range(len(iter_result['Iterations'])):
            temp_sig[:, n] = iter_result['Iterations'][n]['y']
            temp_filters[:, n] = iter_result['Iterations'][n]['f']
        
        CorrMatrix = np.abs(np.corrcoef(temp_sig, rowvar=False))
        CorrMatrix = np.triu(CorrMatrix, 1)
        
        I, J, _ = max_IJ(CorrMatrix)
        Location = (I, J)
        
        XI = temp_sig[:, I] - np.mean(temp_sig[:, I])
        XJ = temp_sig[:, J] - np.mean(temp_sig[:, J])
        
        KI = CK(XI, iter_result['Iterations'][I]['T'])
        KJ = CK(XJ, iter_result['Iterations'][J]['T'])
        
        output = J if KI > KJ else I
        
        # Remove the mode with lower CK
        temp_sig = np.delete(temp_sig, output, axis=1)
        temp_filters = np.delete(temp_filters, output, axis=1)
        iter_result['CorrMatrix'] = CorrMatrix
        iter_result['ComparedModeNum'] = Location
        iter_result['StopNum'] = output
        
        result.append(iter_result)
        
        if temp_filters.shape[1] == ModeNum - 1:
            break
        
        itercount += 1
    
    # Final Mode
    Final_Mode = np.column_stack([iter_result['Iterations'][nn]['y'] for nn in range(len(iter_result['Iterations']))])
    
    return Final_Mode

# -------------------------
# Auxiliary functions
# -------------------------

def TT(y, fs):
    M = fs
    NA = np.correlate(y, y, mode='full') / (np.std(y)**2 * len(y))
    NA = NA[len(NA)//2:]
    
    zeroposi = None
    sample1 = NA[0]
    for lag in range(1, len(NA)):
        sample2 = NA[lag]
        if (sample1 > 0 and sample2 < 0) or sample1 == 0 or sample2 == 0:
            zeroposi = lag
            break
        sample1 = sample2
    
    NA = NA[zeroposi:]
    max_position = np.argmax(NA)
    T = zeroposi + max_position
    return T

def CK(x, T, M=2):
    x = x.flatten()
    N = len(x)
    x_shift = np.zeros((M+1, N))
    x_shift[0, :] = x
    for m in range(1, M+1):
        x_shift[m, T:] = x_shift[m-1, :-T]
    ck = np.sum(np.prod(x_shift, axis=0)**2) / np.sum(x**2)**(M+1)
    return ck

def max_IJ(X):
    tempI = np.argmax(X, axis=0)
    M = np.max(X)
    J = np.argmax(np.max(X, axis=0))
    I = tempI[J]
    return I, J, M

def xxc_mckd(fs, x, f_init, termIter=None, T=None, M=3, plotMode=0):
    """
    Complete Python version of xxc_mckd from MATLAB
    """
    x = x.flatten()
    N = len(x)
    L = len(f_init)
    if termIter is None:
        termIter = 30

    if T is None:
        envelope = np.abs(hilbert(x)) - np.mean(np.abs(hilbert(x)))
        T = round(TT(envelope, fs))
    else:
        T = round(T)
    
    # Initialize filter
    f = f_init.copy()
    ck_best = 0
    y_final = np.zeros((N, termIter))
    f_final = np.zeros((L, termIter))
    ckIter = np.zeros(termIter)
    
    for n in range(termIter):
        # Build XmT matrix
        XmT = np.zeros((L, N, M+1))
        # for m in range(M+1):
        #     for l in range(L):
        #         if l == 0:
        #             XmT[l, m*T:N, m] = x[:N - m*T]
        #         else:
        #             XmT[l, 1:, m] = XmT[l-1, :-1, m]
        for m in range(M+1):
            for l in range(L):
                start_idx = m * T
                if start_idx >= N:
                    continue  # 如果起始索引超过长度，跳过
                if l == 0:
                    length = N - start_idx
                    XmT[l, start_idx:N, m] = x[:length]
                else:
                    length = N - 1
                    if length > 0:
                        XmT[l, 1:N, m] = XmT[l-1, 0:N-1, m]
        
        Xinv = inv(XmT[:, :, 0] @ XmT[:, :, 0].T)
        
        # Compute output signal
        y = (f.T @ XmT[:, :, 0]).T
        
        # Build yt
        yt = np.zeros((N, M+1))
        for m in range(M+1):
            if m == 0:
                yt[:, m] = y
            else:
                yt[m*T:, m] = yt[:N-m*T, m-1]
        
        # Calculate alpha
        alpha = np.zeros((N, M+1))
        for m in range(M+1):
            idx = [i for i in range(M+1) if i != m]
            alpha[:, m] = (np.prod(yt[:, idx], axis=1)**2) * yt[:, m]
        
        # Calculate beta
        beta = np.prod(yt, axis=1)
        
        # Calculate Xalpha
        Xalpha = np.zeros((L,))
        for m in range(M+1):
            Xalpha += XmT[:, :, m] @ alpha[:, m]
        
        # Update filter coefficients
        f = (np.sum(y**2) / (2*np.sum(beta**2))) * (Xinv @ Xalpha)
        f = f / np.sqrt(np.sum(f**2))
        
        # CK for this iteration
        ckIter[n] = np.sum(np.prod(yt, axis=1)**2) / (np.sum(y**2)**(M+1))
        if ckIter[n] > ck_best:
            ck_best = ckIter[n]
        
        # Update T
        envelope = np.abs(hilbert(y)) - np.mean(np.abs(hilbert(y)))
        T = round(TT(envelope, fs))
        
        # Save iteration results
        y_final[:, n] = lfilter(f, 1, x)
        f_final[:, n] = f
    
    return y_final, f_final, ckIter, T

if __name__ == "__main__":
    fs = 20000
    x = np.random.randn(6000)
    # %   fs              :       sampling frequency of x
    # %   x               :       signal to be analyzed
    # %   FilterSiz       :       filter size
    # %   CutNum          :       the cut number of the whole frequency band
    # %   ModeNum         :       the final mode number
    # %   MaxIterNum      :       max iteration number
    #============ FMD 参数 ============#
    filtersize = 30
    cutnum = 7
    modenum = 2
    maxiternum = 20

    mode = FMD(fs, x, filtersize, cutnum, modenum, maxiternum)
    print(mode.shape)