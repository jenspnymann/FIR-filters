# This is my first python project
#if it is yours as well, then run line below
#pip install virtualenv

#To get started do the following 4 lines of powershell commands:
#python -m venv testingenv
#Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
#./env/Scripts/activate.ps1
#pip install -r requirements.txt

import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.signal import fftconvolve, lfilter, firwin
from scipy.signal import convolve as sig_convolve
import numpy as np
from math import pi

msg = "Hello World"
print(msg)
msg1 = msg.upper()
print(msg1)


plt.close('all')

# Design a IRR Butterworth filter
# Fs = 1000

# n = 5
# fc = 100

# w_c = 2*fc/Fs # normalized frequency
# [b,a] = sig.butter(n, w_c)

# # Frequency response lowpass
# [w,h] = sig.freqz(b, a, worN = 100)
# w = Fs*w/(2*pi)

# h_db = 20*np.log10(abs(h))

# # Frequency response bandpass
# fc_bp = np.array([100, 300])
# w_c_bp = 2*fc_bp/Fs # normalized frequency
# [b,a] = sig.butter(n, w_c_bp, btype='bandpass')

# [w,h] = sig.freqz(b, a, worN = 100)
# w = Fs*w/(2*pi)

# h_db = 20*np.log10(abs(h))


# plt.figure()
# plt.plot(w, h_db); plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude(dB)')
# plt.grid('on')
# plt.show()
# print('DONE IIR')

# Design FIR filter
N = 100         # number of samples
Fs = 1000       # sampling frequency remember nyquist says more than 2 times bandwidth or carrier frequency and so on
fc = 100        # -3db cut off frequency
w_c = 2*fc/Fs
b = sig.firwin(N,w_c)

[w,h] = sig.freqz(b, worN=100)
w = Fs*w/(2*pi)

h_db = 20*np.log(abs(h))


print('Showing FIR filter')
plt.figure()
plt.plot(w, h_db)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude(dB)')
plt.title('FIR filter response')
plt.grid('on')
plt.show()
print('DONE FIR')

# now for the noise
N_array = list(range(1,101))
duration_of_samples = N  # sampling rate [Hz] times number of samples
duration_of_seconds = N/Fs  # sampling rate [Hz] times number of samples
white_noise = np.random.default_rng().uniform(-1, 1, N)
plt.figure()
plt.plot(N_array, white_noise)
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('white noise')
plt.grid('on')
plt.show()

# try send a impulse into the FIR filter
impulse = [1]+[0]*99    # [1, 0, 0, 0, 0, 0, 0, 0... 0]
response = lfilter(b, [1.0], impulse)
print('response')
print(response)
N_array = list(range(1,101))
duration_of_samples = N  # sampling rate [Hz] times number of samples
duration_of_seconds = N/Fs  # sampling rate [Hz] times number of samples
white_noise = np.random.default_rng().uniform(-1, 1, N)
plt.figure()
plt.plot(N_array, response)
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('response')
plt.grid('on')
plt.show()
print('b')
print(b)
h_n_test = [1, 2, 2, 4]
x_n_test = [1, 2, 2]
response_test = lfilter(h_n_test, [1.0], x_n_test) # 1, 4, 8 (see below for equation)
# response is x[0]*h[0], x[1]*h[0]+x[0]*h[1], X[2]*h[0]+x[1]*h[1]+x[0]*h[2]
print('response_test')
print(response_test)
print('DONE FIR response')
