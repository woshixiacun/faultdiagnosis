import numpy as np
from scipy.signal import firwin, hann, hilbert, lfilter, correlate
from numpy.linalg import inv

from xxc_mckd import xxc_mckd
from utils import TT, CK, max_IJ


def FMD(fs, x, FilterSize, CutNum, ModeNum, MaxIterNum):
    x = np.array(x).reshape(-1, 1)
    freq_bound = np.arange(0, 1, 1 / CutNum) # length=7

    # 构造滤波器
    temp_filters = np.zeros((FilterSize, CutNum)) # 一列是一个filter 矩阵 FilterSize x CutNum
    for i, fb in enumerate(freq_bound):
        # hann() 生成一个 长度为 FilterSize 的 Hanning 窗
            # 返回值 w 是一个 一维 NumPy 数组，元素值在 0~1 之间. 
            # 数组 w 就是窗函数，在滤波或频谱分析时用来加权信号，减小频谱泄漏或滤波器的旁瓣。
        # firwin() 是 SciPy 里的 FIR 滤波器设计函数（Window Method）：  (原matlab代码中也是用的工具箱)
            # 返回一个 FIR 滤波器系数数组，长度 = FilterSize
            # window=w → 使用你之前定义的 Hanning 窗函数
            # pass_zero=False → 设计 带通或带阻滤波器，而不是低通
            # cutoff（这里 [fb + 1e-6, fb + 1/CutNum - 1e-6]） → 截止频率或频带
        # w = hann(FilterSize) 
        # temp_filters[:, i] = firwin(FilterSize, 
        #                             [fb + 1e-6, fb + 1/CutNum - 1e-6],
        #                             window=w, pass_zero=False)

        temp_filters[:, i] = firwin(FilterSize,
                            [fb + 1e-6, fb + 1/CutNum - 1e-6],  # 生成具有截止频率区间[fl,fu]的个 FIR 滤波器
                            window='hann', pass_zero=False)  # 每次初始化一列
        

        # np.tile 的作用
            # np.tile(A, reps) 用来 重复数组 A，生成更大的数组
            # reps 是重复次数，可以是整数或元组，表示每个维度的重复次数
    temp_sig = np.tile(x, (1, CutNum)) #原始信号重复7次
    itercount = 2
    result = {}

    while True:
        iternum = 2 if itercount != 2 else MaxIterNum - (CutNum - ModeNum) * 2 #
        result[itercount] = {"Iter": [], "Signal": [], "Filter": [], "Freq": [], "T": []}
        # result(1, :) = {'IterCount', 'Iterations', 'CorrMatrix', 'ComparedModeNum', 'StopNum'};
        for n in range(temp_filters.shape[1]):
            f_init = temp_filters[:, n]
            y_iter, f_iter, k_iter, T_iter = xxc_mckd(fs, temp_sig[:, n], f_init, iternum, None, 1, 0)

            result[itercount]["Signal"].append(y_iter[:, -1])
            result[itercount]["Filter"].append(f_iter[:, -1])
            result[itercount]["T"].append(T_iter)

        # 更新信号
        for n in range(len(result[itercount]["Signal"])):
            temp_sig[:, n] = result[itercount]["Signal"][n]
            temp_filters[:, n] = result[itercount]["Filter"][n]

        CorrMatrix = np.abs(np.corrcoef(temp_sig.T))
        CorrMatrix = np.triu(CorrMatrix, 1)

        I, J, _ = max_IJ(CorrMatrix)
        XI = temp_sig[:, I] - np.mean(temp_sig[:, I])
        XJ = temp_sig[:, J] - np.mean(temp_sig[:, J])

        KI = CK(XI, result[itercount]["T"][I], 1)
        KJ = CK(XJ, result[itercount]["T"][J], 1)
        output = J if KI > KJ else I  # 剔除相关性大的一组

        temp_sig = np.delete(temp_sig, output, axis=1)
        temp_filters = np.delete(temp_filters, output, axis=1)

        if temp_filters.shape[1] == ModeNum - 1:
            break

        itercount += 1

    Final_Mode = np.column_stack(result[itercount]["Signal"])
    return Final_Mode



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