import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from FMD import FMD                # ⬅ 前面已给你 Python 转写的 FMD 主函数
                                  # xxc_mckd/TT/CK/max_IJ 也应在同文件或单独导入

#---------------------- FFT 函数替代 myfft.m ----------------------#
def myfft(fs, x, plot_flag=1):
    N = len(x)
    X = np.fft.fft(x) # 使用 NumPy 的 快速傅里叶变换（FFT），将时域信号 x 转换到频域, 输出 X 是复数数组（复数包含幅度和相位信息）
    freq = np.linspace(0, fs/2, N//2) # 构建 频率轴
    amp = np.abs(X[:N//2]) * 2 / N
 
    if plot_flag:
        plt.plot(freq, amp)
    return freq, amp


#==================================================================#
#                 主 试 例 复 现 MATLAB 脚 本
#==================================================================#
# ① 载入信号 x.mat，其中应包含变量 x
data = sio.loadmat('x.mat')
x = np.array(data['x']).squeeze()     # MATLAB 1列→python一维数组
fs = 20000
t = np.arange(len(x)) / fs


#============ 波形 ============#
plt.figure("Time waveform of mixed signal")
plt.plot(t, x, 'b')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.savefig("mixed_signal.png", dpi=300, bbox_inches='tight')  # 保存图片


#============ 原信号 FFT ============#
plt.figure("FFT amplitude spectrum of mixed signal")
myfft(fs, x, plot_flag=1)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")


#============ FMD 参数 ============#
filtersize = 30
cutnum = 7
modenum = 2
maxiternum = 20

print("Running FMD ...")
y_final = FMD(fs, x, filtersize, cutnum, modenum, maxiternum)
print("Done.")


#============ 分量时域图 ============#
b = y_final.shape[1]
plt.figure("Time waveform of mode(s)")
for k in range(b):
    plt.subplot(b,1,k+1)
    plt.plot(t, y_final[:,k],'b')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")


#============ 分量 FFT ============#
plt.figure("FFT amplitude spectrum of mode(s)")
for k in range(b):
    plt.subplot(b,1,k+1)
    ff, amp = myfft(fs, y_final[:,k], plot_flag=0)
    plt.plot(ff, amp/np.max(amp), 'r')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Norm Amp")


#============ Hilbert 包络谱 ============#
plt.figure("Hilbert envelope spectrum of mode(s)")
for k in range(b):
    plt.subplot(b,1,k+1)
    env = np.abs(hilbert(y_final[:,k])) - np.mean(np.abs(hilbert(y_final[:,k])))
    ff, amp = myfft(fs, env, plot_flag=0)
    plt.plot(ff, amp,'b')
    plt.xlim([0,300])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amp")


plt.show()
