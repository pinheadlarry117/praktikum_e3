import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks

fs, x = wavfile.read("distorted.wav")
x = x.astype(float)

t = np.arange(len(x)) / fs

plt.figure()
plt.plot(t, x)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Time-domain signal")
plt.show()


N = len(x)
X = np.fft.fft(x)

# Frequency axis
f = np.fft.fftfreq(N, d=1/fs)

# Keep positive frequencies only
pos = f >= 0
f_pos = f[pos]
X_pos = X[pos]

plt.figure()
plt.plot(f_pos, np.abs(X_pos))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("Magnitude spectrum of distorted signal")
plt.xlim(0, 4000)  # Nyquist limit
plt.grid(True)
plt.show()

plt.figure()
plt.plot(f_pos, np.abs(X_pos))
plt.xlim(0, 2000)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

"""
optional
N_fft = 8 * N   # zero-padding
Xz = np.fft.fft(x, n=N_fft)
fz = np.fft.fftfreq(N_fft, d=1/fs)

posz = fz >= 0

plt.figure()
plt.plot(fz[posz], np.abs(Xz[posz]))
plt.xlim(0, 4000)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("Zero-padded magnitude spectrum")
plt.grid(True)
plt.show()


peaks, _ = find_peaks(np.abs(X_pos), height=np.max(np.abs(X_pos))*0.2)

frequencies = f_pos[peaks]
magnitudes = np.abs(X_pos[peaks])

for f0, mag in zip(frequencies, magnitudes):
    print(f"Frequency ≈ {f0:.1f} Hz, Magnitude ≈ {mag:.2f}")

"""