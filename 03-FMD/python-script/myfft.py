import numpy as np
import matplotlib.pyplot as plt

def myfft(fs, x, plotMode=0):
    """
    Fourier Spectrum of x.

    Parameters
    ----------
    fs : float
        Sampling frequency.
    x : array_like
        Input signal.
    plotMode : int, optional
        0 = no plot, 1 = plot spectrum. Default = 0.

    Returns
    -------
    ff : array
        Frequency axis.
    amp : array
        Amplitude spectrum.
    """

    x = np.asarray(x).flatten()
    NN = len(x)

    # Frequency axis
    ff = np.linspace(0, fs, NN+1)[:NN]

    # Spectrum (2/N norm same to MATLAB)
    amp = np.abs(np.fft.fft(x) / NN * 2)
    amp = amp[:NN//2]
    ff = ff[:NN//2]

    if plotMode == 1:
        plt.plot(ff, amp)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")

    return ff, amp


if __name__ == "__main__":
    fs = 12000
    x = np.random.randn(6000)
    ff, amp = myfft(fs, x, plotMode=1)
