import scipy, pylab
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread


def audioreader(path):
    [fs, rowdata] = wavread(path)  # x is a numpy array of integer, representing the samples
    # scale to -1.0 -- 1.0
    if rowdata.dtype == 'int16':
        nb_bits = 16  # -> 16-bit wav files
    elif rowdata.dtype == 'int32':
        nb_bits = 32  # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    rowsamples = rowdata / (max_nb_bit + 1.0)  # samples is a numpy array of float representing the samples
    # get all channels
    samples = []
    channelsz = rowsamples[0].__len__()
    for i in range(channelsz):
        samples.append(rowsamples[:, i])
    #print(samples)
    return fs, np.array(samples)


# short time fourier transform
# x: data array
# fs: sampling rate
# framesz: frame size
# overlap: overlap
def mystft(x, fs, framesz, overlap):
    framesamp = int(framesz * fs)
    overlapsamp = int(overlap * fs)
    step = int(1 / framesz)
    # use hanning window
    w = scipy.hanning(framesamp)
    X = []
    F = np.arange(0, fs, step)
    T = []
    tcout = 0
    for i in range(0, len(x) - framesamp, overlapsamp):
        X.append(scipy.fft(w * x[i:i + framesamp]))
        T.append(tcout * overlap)
        tcout += 1
    return np.array(X), F, np.array(T)


# inverse short time fourier transform
# X: frequency-time matrix
# fs: sampling rate
# size: original audio data size
# framesz: frame size
# overlap: overlap
def myistft(X, fs, size, framesz, overlap):
    x = np.zeros(size)
    offset = 0
    framesamp = int(framesz * fs)
    overlapsamp = int(overlap * fs)
    for i in range(X.__len__()):
        if offset + framesamp <= size:
            x[offset:offset + framesamp] += scipy.real(scipy.ifft(X[i]))
            offset += overlapsamp
    return x


if __name__ == '__main__':
    # read wav file data
    fs, data = audioreader("test.wav")
    target = data[0]
    dataSz = target.__len__()
    plt.plot(target)
    plt.show()

    # Create test signal and STFT
    frameSz = 0.050  # with a frame size of 50 milliseconds
    hop = 0.025  # and hop size of 25 milliseconds.
    X, F, T = mystft(target, fs, frameSz, hop)
    #print(X)

    # deal with Y to absolute number and transformed by log10
    Y = scipy.absolute(X)
    # for line in Y:
    #     for col in line:
    #         col = math.log10(col)

    # Plot the magnitude spectrogram.
    plt.pcolormesh(T, F.T, Y.T, vmin=0, vmax=2*np.sqrt(2))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    # Compute the ISTFT.
    xhat = myistft(X, fs, dataSz, frameSz, hop)

    plt.plot(xhat)
    plt.show()
