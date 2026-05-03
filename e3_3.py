import numpy as np
import matplotlib.pyplot as plt

N = 64
k = np.arange(N)

x0 = np.zeros(N)
x1 = np.zeros(N)
x2 = np.zeros(N)

x0[0] = 1
x1[1] = 1
x2[2] = 1

def show_fft(x, title):
    X = np.fft.fft(x)

    plt.figure(figsize=(10, 8))

    plt.subplot(4, 1, 1)
    plt.stem(np.real(X))
    plt.title(title + " - Real Part")

    plt.subplot(4, 1, 2)
    plt.stem(np.imag(X))
    plt.title("Imaginary Part")

    plt.subplot(4, 1, 3)
    plt.stem(np.abs(X))
    plt.title("Magnitude")

    plt.subplot(4, 1, 4)
    plt.stem(np.angle(X))
    plt.title("Phase")

    plt.tight_layout()
    plt.show()

show_fft(x0, "δ(k)")
show_fft(x1, "δ(k-1)")
show_fft(x2, "δ(k-2)")

"""
Magnitude: constant for all frequencies
→ abs(X(Ω)) = 1
A time shift produces a linear phase change but does not affect magnitude.
"""

n = 5
Omega0 = 2 * np.pi * n / N
x_bin = np.cos(Omega0 * k)

show_fft(x_bin, "Cosine - integer DFT bin")

"""
Two sharp peaks at frequency bins ±n\pm n±n
Imaginary part ≈ 0
Phase is either 0 or π
No spectral leakage
Perfect frequency alignment with FFT bins
"""

Omega1 = 2 * np.pi * 5.5 / N
x_nobin = np.cos(Omega1 * k)

show_fft(x_nobin, "Cosine - non-integer DFT bin")

"""
Energy spreads over many bins
→ Spectral leakage
Both real and imaginary parts are non-zero
Phase becomes irregular
Cause: FFT assumes periodicity with period NNN; frequency mismatch breaks orthogonality.
"""