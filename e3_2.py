import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

fs, x = wavfile.read("voice1.wav")
x = x.astype(float)   # convert to float

N = len(x)
duration_ms = N / fs * 1000
print("Length of signal:", duration_ms, "ms")

rxx = np.correlate(x, x, mode="full")
rxx = rxx[rxx.size // 2:]

lags_ms = np.arange(len(rxx)) / fs * 1000

plt.figure()
plt.plot(lags_ms, rxx)
plt.xlabel("Lag (ms)")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation of Speech Signal")
plt.show()

min_lag = int(fs * 0.002)  # ignore <2 ms
peak_index = np.argmax(rxx[min_lag:]) + min_lag

pitch_period_ms = peak_index / fs * 1000
fundamental_freq = fs / peak_index

print("Pitch period:", pitch_period_ms, "ms")
print("Fundamental frequency:", fundamental_freq, "Hz")

data = np.load("sequences.npz")

x  = data["x"]
y1 = data["y1"]
y2 = data["y2"]
y3 = data["y3"]

def norm_corr(a, b):
    return np.correlate(a, b, mode="full") / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

c1 = norm_corr(x, y1)
c2 = norm_corr(x, y2)
c3 = norm_corr(x, y3)

def find_shift_and_peak(c):
    idx = np.argmax(np.abs(c))
    shift = idx - (len(x) - 1)
    peak = c[idx]
    return shift, peak

shift1, peak1 = find_shift_and_peak(c1)
shift2, peak2 = find_shift_and_peak(c2)
shift3, peak3 = find_shift_and_peak(c3)

print("y1: shift =", shift1, "peak =", peak1)
print("y2: shift =", shift2, "peak =", peak2)
print("y3: shift =", shift3, "peak =", peak3)

def scaling_factor(x, y):
    return np.dot(y, x) / np.dot(x, x)

                                 
a1 = scaling_factor(x, y1)                               
a2 = scaling_factor(x, y2)
a3 = scaling_factor(x, y3)

print("Scaling y1:", a1)
print("Scaling y2:", a2)
print("Scaling y3:", a3)

"""
shift:
A large correlation peak
NOT centered at zero lag
shift = peak_index - (len(x) - 1)
scaled:
Has a large correlation peak
Scaling factor ≠ 1
a ≈ 1 no scaling
a > 1 amplification
a < 1 attenuation
"""