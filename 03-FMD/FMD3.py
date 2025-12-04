import numpy as np
from scipy import signal
from scipy.linalg import inv
from scipy.signal import hilbert

# 辅助函数 1: 计算 Correlated Kurtosis (CK)
def CK(x, T, M=2):
    """
    计算相关峭度 (Correlated Kurtosis, CK)
    :param x: 信号 (numpy 1D array)
    :param T: 估计的周期 (样本数)
    :param M: 阶数 (默认 2)
    :return: CK 值
    """
    x = x.flatten()
    N = len(x)
    
    # x_shift (M+1) x N matrix
    x_shift = np.zeros((M + 1, N))
    x_shift[0, :] = x
    
    # 模拟 MATLAB 的循环移位
    for m in range(1, M + 1):
        if T * m < N:
            # Python: [T:] is equivalent to MATLAB's (T+1):end
            # Python: [:-T] is equivalent to MATLAB's 1:end-T
            x_shift[m, T:] = x_shift[m - 1, :-T]

    # prod(x_shift) 是对每一列的元素进行乘积，得到一个长度为 N 的行向量
    # sum(prod(x_shift).^2)
    numerator = np.sum(np.prod(x_shift, axis=0)**2)
    
    # sum(x.^2)^(M + 1)
    denominator = np.sum(x**2)**(M + 1)
    
    if denominator == 0:
        return 0.0

    ck = numerator / denominator
    return ck

# 辅助函数 2: 基于自相关函数估计周期 T
def TT(y, fs):
    """
    使用自相关函数估计信号 y 的周期 T (样本数)
    :param y: 信号包络 (已减去均值)
    :param fs: 采样频率
    :return: 估计的周期 T (样本数)
    """
    y = y.flatten()
    N = len(y)
    M = int(fs) # max lag

    # 计算自相关函数 (mode='full' 结果长度 2*N-1)
    # MATLAB xcorr(y, y, M, 'coeff')
    NA_full = np.correlate(y, y, mode='full')
    
    # 归一化: MATLAB 'coeff' 使得 lag 0 处的自相关值为 1
    if NA_full[N - 1] == 0:
        return M # 无法计算，返回最大延迟
        
    NA_full_norm = NA_full / NA_full[N - 1]
    
    # 选取后半部分 (lag 0 到 N-1), 并限制最大 lag M
    # MATLAB: NA(ceil(length(NA) / 2):end) -> Python: NA_full_norm[N - 1:]
    NA_full_half = NA_full_norm[N - 1:]
    NA = NA_full_half[:M + 1] 
    
    # 寻找第一个零交叉点
    sample1 = NA[0] # lag 0 的值
    zeroposi = -1

    # 循环从 lag 1 开始 (Python index 1)
    for lag in range(1, len(NA)):
        sample2 = NA[lag]
        
        # 零交叉或恰好过零点的逻辑: (正到负) 或 (任一为零)
        # (sample1 > 0 && sample2 < 0) || (sample1 == 0 || sample2 == 0)
        if (sample1 > 0 and sample2 < 0) or (sample1 == 0 or sample2 == 0):
             # zeroposi 存储的是 0-based index/lag
             zeroposi = lag
             break
        
        sample1 = sample2
    
    if zeroposi == -1:
         # 如果在 M 内未找到零交叉，则返回 M
         return M
        
    # 从零交叉点处截断
    NA_cut = NA[zeroposi:]
    
    # 找到截断后信号的最大值的位置 (相对位置)
    max_position_relative = np.argmax(NA_cut)
    
    # 估计的周期 T: 零交叉点的 lag + 相对最大值的 lag
    # T = zeroposi + max_position (MATLAB logic for 1-based index is the same as 0-based lag)
    T = zeroposi + max_position_relative

    # T 已经是样本数，因为 lag 是样本数
    return T

# 辅助函数 3: 最大化相关峭度去卷积 (MCKD/XXC-MCKD 迭代)
def xxc_mckd(fs, x, f_init, termIter, T_in, M_in, plotMode):
    """
    实现了论文中 FMD 内部使用的 XXC-MCKD 迭代过程，用于滤波器更新。
    :param fs: 采样频率
    :param x: 输入信号
    :param f_init: 初始滤波器系数
    :param termIter: 最大迭代次数
    :param T_in: 初始周期 (如果为空，则计算)
    :param M_in: 阶数 M
    :param plotMode: 绘图模式 (0: 不绘图)
    :return: 滤波信号, 最终滤波器, CK迭代值, 最终周期
    """
    # 处理默认值
    termIter = termIter if termIter is not None else 30
    plotMode = plotMode if plotMode is not None else 0
    M = M_in if M_in is not None else 3
    
    x = x.flatten()
    L = len(f_init) # 滤波器长度
    N = len(x)      # 信号长度
    
    # 如果 T_in 为空，则进行周期估计
    if T_in is None or len(T_in) == 0:
        # abs(hilbert(x)) - mean(abs(hilbert(x)))
        xxenvelope = np.abs(hilbert(x)) - np.mean(np.abs(hilbert(x)))
        T = TT(xxenvelope, fs)
    else:
        T = T_in

    T = int(np.round(T))
    if T == 0: T = 1 # 周期不能为 0

    # ----------------------------
    # 初始 XmT 计算 (L x N x (M+1))
    XmT = np.zeros((L, N, M + 1))
    
    # XmT 矩阵的计算
    for m in range(M + 1):
        if m * T < N:
            # l=0 (MATLAB l=1): XmT[0, m*T:end, m] = x[1:N - m*T]
            XmT[0, m * T:, m] = x[:N - m * T]
        
        # l=1 to L-1 (MATLAB l=2 to L)
        for l in range(1, L):
            # XmT[l, 2:end, m] = XmT[l - 1, 1:end - 1, m]
            XmT[l, 1:, m] = XmT[l - 1, :-1, m]

    # Xinv = inv(XmT(:, :, 1) * XmT(:, :, 1)')
    X_0 = XmT[:, :, 0] # 对应于 m=0 的 XmT
    Xinv = inv(X_0 @ X_0.T)

    f = f_init.reshape(-1, 1) # 确保 f 是列向量
    ck_best = 0.0

    # 初始化迭代矩阵
    y_Iter = np.zeros((N, termIter))
    f_Iter = np.zeros((L, termIter))
    ckIter = np.zeros(termIter)
    
    n = 0 # 迭代计数器 (0-based)

    while n < termIter:
        # Compute output signal: y = (f' * XmT(:, :, 1))'
        # y 是 N x 1 向量
        y = (f.T @ X_0).T 

        # Generate yt (y 及其 T 周期移位)
        f_Iter[:, n] = f.flatten()
        yt = np.zeros((N, M + 1))
        
        for m in range(M + 1):
            if m == 0:
                yt[:, m] = y.flatten()
            else:
                # yt[T:, m] = yt[:-T, m-1]
                if T * m < N:
                    # 递归计算 yt: yt[:, m] 是 y 延迟 m*T 的版本
                    yt[T:, m] = yt[:-T, m-1] # 这里的 T 应该是当前的 T

        # Calculate alpha
        alpha = np.zeros((N, M + 1))
        for m in range(M + 1):
            # 找到除了第 m 列以外的所有列
            other_cols_indices = [i for i in range(M + 1) if i != m]
            
            # prod(yt(:, [1:m (m + 2):size(yt, 2)]), 2)
            # 对应于 yt 中除了 m 以外的列的乘积 (按行)
            prod_others = np.prod(yt[:, other_cols_indices], axis=1)
            
            # alpha(:, m + 1) = (prod(...)**2) .* yt(:, m + 1)
            alpha[:, m] = (prod_others**2) * yt[:, m]

        # Calculate beta
        # beta = prod(yt, 2)
        beta = np.prod(yt, axis=1).reshape(-1, 1) # 确保 beta 是 N x 1 列向量

        # Calculate the sum Xalpha term
        # Xalpha 是 L x 1 列向量
        Xalpha = np.zeros((L, 1))
        for m in range(M + 1):
            # Xalpha = Xalpha + XmT(:, :, m + 1) * alpha(:, m + 1)
            Xalpha += XmT[:, :, m] @ alpha[:, m].reshape(-1, 1)

        # Calculate the new filter coefficients
        # f = sum(y.^2) / (2 * sum(beta.^2)) * Xinv * Xalpha;
        if np.sum(beta**2) == 0:
             # 如果分母为 0，则停止迭代，使用当前的 f
             f_Iter[:, n] = f.flatten()
             y_Iter[:, n] = (f.T @ X_0).flatten()
             ckIter[n] = 0.0
             break

        f = (np.sum(y**2) / (2 * np.sum(beta**2))) * (Xinv @ Xalpha)
        
        # Normalize: f = f / sqrt(sum(f.^2));
        f_norm = np.linalg.norm(f)
        if f_norm != 0:
             f = f / f_norm

        # Calculate the ck value of this iteration
        # ckIter(n) = sum(prod(yt, 2).^2) / (sum(y.^2)^(M + 1));
        ckIter[n] = np.sum(beta.flatten()**2) / (np.sum(y**2)**(M + 1))
        
        # Update the best match filter if required
        if ckIter[n] > ck_best:
            ck_best = ckIter[n]

        # Update the period T (自适应 T 估计)
        # xyenvelope = abs(hilbert(y)) - mean(abs(hilbert(y)));
        xyenvelope = np.abs(hilbert(y.flatten())) - np.mean(np.abs(hilbert(y.flatten())))
        T = TT(xyenvelope, fs)
        T = int(np.round(T))
        if T == 0: T = 1

        # Re-calculate XmT and Xinv with the new T
        XmT = np.zeros((L, N, M + 1))
        for m in range(M + 1):
            if m * T < N:
                XmT[0, m * T:, m] = x[:N - m * T]
            
            for l in range(1, L):
                XmT[l, 1:, m] = XmT[l - 1, :-1, m]

        X_0 = XmT[:, :, 0]
        # 检查矩阵是否可逆
        try:
             Xinv = inv(X_0 @ X_0.T)
        except np.linalg.LinAlgError:
             # 如果不可逆，则停止迭代
             f_Iter[:, n] = f.flatten()
             y_Iter[:, n] = (f.T @ X_0).flatten()
             break

        # 最终信号为当前滤波器对原始信号的滤波
        # y_final(:, n) = filter(f_final(:, n), 1, x);
        y_Iter[:, n] = signal.lfilter(f.flatten(), 1, x)

        n += 1
    
    # 截断到实际迭代次数
    y_final = y_Iter[:, :n]
    f_final = f_Iter[:, :n]
    ckIter = ckIter[:n]
    T_final = T # T is the last calculated period

    return y_final, f_final, ckIter, T_final

# 辅助函数 4: 找到矩阵上三角的最大值及其 1-based 索引
def max_IJ(X):
    """
    找到矩阵 X 的上三角部分 (对角线以上) 的最大值及其 1-based 索引 I 和 J。
    """
    # X 已经是上三角矩阵 (np.triu(..., k=1))
    
    # 1. 找到每列的最大值 (temp) 及其行索引 (tempI)
    temp = np.amax(X, axis=0)
    tempI = np.argmax(X, axis=0) # Row indices (0-based)
    
    # 2. 找到列最大值中的最大值 (M) 及其列索引 (J)
    J = np.argmax(temp) # Column index (0-based)
    M = temp[J]        # Max value
    I = tempI[J]       # Row index (0-based)
    
    # 返回 MATLAB 风格的 1-based 索引
    I_matlab = I + 1
    J_matlab = J + 1
    
    return I_matlab, J_matlab, M

# 主函数: Feature Mode Decomposition (FMD)
def FMD(fs, x, FilterSize, CutNum, ModeNum, MaxIterNum):
    """
    Feature Mode Decomposition (FMD) 主函数
    :param fs: 采样频率
    :param x: 待分析信号
    :param FilterSize: 滤波器长度
    :param CutNum: 整个频带的切分数量 (初始化滤波器的数量)
    :param ModeNum: 最终的模态数量
    :param MaxIterNum: 最大迭代次数
    :return: 最终分解出的模态 (Final_Mode)
    """
    x = x.flatten()
    N = len(x)

    # ------------------------------------
    # 初始化 (Initialization)
    # ------------------------------------
    
    # freq_bound = 0:1/CutNum:1 - 1/CutNum;
    freq_bound = np.arange(0, 1.0, 1.0 / CutNum)
    
    temp_filters = np.zeros((FilterSize, CutNum))
    window = signal.windows.hann(FilterSize)
    numtaps = FilterSize # MATLAB fir1 order is FilterSize - 1

    eps = np.finfo(float).eps
    
    # 构建 Hanning 窗滤波器组
    for n in range(CutNum):
        # [freq_bound(n) + eps, freq_bound(n) + 1 / CutNum - eps]
        low_cut = freq_bound[n] + eps
        high_cut = freq_bound[n] + 1.0 / CutNum - eps
        
        # fir1(FilterSize - 1, [low, high], w)'
        # fs=2.0 (Nyquist=1.0) 使得 cutoff 直接使用归一化频率
        # scale=False 避免增益影响
        f = signal.firwin(numtaps, [low_cut, high_cut], window=window, pass_zero='bandpass', scale=False, fs=2.0)
        
        temp_filters[:, n] = f.T

    # 结果初始化 (使用列表模拟 MATLAB 的 cell array)
    # result(1, :) = {'IterCount', 'Iterations', 'CorrMatrix', 'ComparedModeNum', 'StopNum'};
    result = [['IterCount', 'Iterations', 'CorrMatrix', 'ComparedModeNum', 'StopNum']]

    # ------------------------------------
    # 迭代和模态选择 (Iterations and Mode Selection)
    # ------------------------------------
    
    # temp_sig = repmat(x, [1, CutNum]);
    temp_sig = np.tile(x.reshape(-1, 1), (1, CutNum)) # N x CutNum
    
    itercount = 1 # 1-based iteration counter (index 0 is header)
    
    while True:
        itercount += 1 
        iternum = 2
        
        # MATLAB: if itercount == 2; iternum = MaxIterNum - (CutNum - ModeNum) * iternum; end
        if itercount == 2:
            iternum = MaxIterNum - (CutNum - ModeNum) * iternum
        
        current_result_row = [iternum, None, None, None, None]
        
        # Iterations: 针对每个滤波器进行 MCKD 迭代
        # result{itercount, 2} 是一个包含 CutNum 个元素的 list of lists
        X = [] 
        for n in range(temp_filters.shape[1]): # temp_filters.shape[1] is the current number of filters
            f_init = temp_filters[:, n].reshape(-1, 1) # 传入列向量
            
            # xxc_mckd 返回 y_Iter(N x termIter), f_Iter(L x termIter), ckIter(termIter), T_Iter
            y_Iter, f_Iter, k_Iter, T_Iter = xxc_mckd(fs, temp_sig[:, n], f_init, iternum, None, 1, 0)
            
            # result{itercount, 2}{n, 1} = y_Iter(:, end) # 最终滤波信号
            # result{itercount, 2}{n, 2} = f_Iter(:, end) # 最终滤波器系数
            # result{itercount, 2}{n, 3} = k_Iter(:, end) # 最终 CK 值
            # result{itercount, 2}{n, 6} = T_Iter         # 最终周期
            
            # 计算振幅谱峰值频率
            fft_f = np.abs(np.fft.fft(f_Iter[:, -1]))
            # 只需要前半部分 (0 到 Nyquist)
            fft_f_half = fft_f[:FilterSize // 2]
            index1 = np.argmax(fft_f_half)
            center_freq = (index1) * (fs / FilterSize)
            
            # 存储 (y_final, f_final, ck_final, fft_f_half, center_freq, T_final)
            X.append([y_Iter[:, -1].reshape(-1, 1), f_Iter[:, -1].reshape(-1, 1), k_Iter[-1], fft_f_half.reshape(-1, 1), center_freq, T_Iter])
        
        current_result_row[1] = X
        
        # 更新 temp_sig 和 temp_filters 以进行模态选择
        temp_sig = np.hstack([X[n][0] for n in range(len(X))]) # N x CurrentCutNum
        temp_filters = np.hstack([X[n][1] for n in range(len(X))]) # L x CurrentCutNum
        
        # 模态选择 (Mode Selection)
        
        # CorrMatrix = abs(corrcoef(temp_sig))
        # rowvar=False 表示每列是一个变量 (信号)
        CorrMatrix = np.abs(np.corrcoef(temp_sig, rowvar=False))
        
        # CorrMatrix = triu(CorrMatrix, 1)
        # k=1 表示上三角，对角线为 0
        CorrMatrix = np.triu(CorrMatrix, k=1) 
        
        # [I, J, ~] = max_IJ(CorrMatrix); (I, J 是 1-based 索引)
        I, J, M_corr = max_IJ(CorrMatrix)
        
        # 转换为 0-based 索引
        I_idx, J_idx = I - 1, J - 1
        
        # 计算 Correlated Kurtosis (CK)
        XI = X[I_idx][0].flatten()
        XJ = X[J_idx][0].flatten()
        
        # 减去均值
        XI = XI - np.mean(XI)
        XJ = XJ - np.mean(XJ)
        
        T_I = X[I_idx][5]
        KI = CK(XI, T_I, 1) # M=1, 对应于论文中的简化版 CK
        
        T_J = X[J_idx][5]
        KJ = CK(XJ, T_J, 1)
        
        # 丢弃 CK 值较小的模态
        if KI > KJ:
            output_idx = J_idx # 丢弃 J
        else:
            output_idx = I_idx # 丢弃 I
            
        output = output_idx + 1 # 1-based 索引
        
        # 从 X, temp_sig, temp_filters 中删除冗余模态
        X.pop(output_idx)
        temp_sig = np.delete(temp_sig, output_idx, axis=1)
        temp_filters = np.delete(temp_filters, output_idx, axis=1)
        
        current_result_row[2] = CorrMatrix
        current_result_row[3] = (I, J)
        current_result_row[4] = output
        
        result.append(current_result_row)
        
        # 检查终止条件: size(temp_filters, 2) == ModeNum-1
        # 当前剩余滤波器数量为 ModeNum
        if temp_filters.shape[1] == ModeNum:
            break
        
    # ------------------------------------
    # 输出 (Output)
    # ------------------------------------
    
    # 最终结果存储在最后一个 result 元素的 X (即 current_result_row[1]) 中
    X_final = result[-1][1] 
    
    Final_Mode = np.zeros((N, ModeNum))
    
    # 提取最终模态
    for nn in range(ModeNum):
        # X_final[nn][0] 是最终滤波信号 y_Iter[:, end]
        Final_Mode[:, nn] = X_final[nn][0].flatten()

    return Final_Mode

# --- 示例用法 (用于测试): 请替换为您的实际数据 ---
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
