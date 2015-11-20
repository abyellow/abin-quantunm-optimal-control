import numpy as np
import matplotlib.pyplot as plt


def plotSpec(y,Fs):
 """
 Plots a Single-Sided Amplitude Spectrum of y(t)
 """
 n = len(y) # length of the signal
 k = np.array(range(n))
 T = n/Fs
 frq = k/T # two sides frequency range
 frq = np.fft.fftfreq[np.array(range(n/2))] # one side frequency range

 Y = np.fft.rfft(y)/n # fft computing and normalization
 Y = Y[np.array(range(n/2))]
 
 plt.plot(frq,abs(Y),'r') # plotting the spectrum
 plt.xlabel('Freq (Hz)')
 plt.ylabel('|Y(freq)|')




