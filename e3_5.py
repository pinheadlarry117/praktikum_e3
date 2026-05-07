import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import sounddevice as sd

fs, x = wavfile.read("voice2.wav")
x = x.astype(float)

sd.play(x, fs)

f, t, Sxx = spectrogram(x, fs)

plt.figure()
plt.pcolormesh(t, f, 10*np.log10(Sxx), shading="gouraud")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.title("Default Spectrogram")
plt.colorbar(label="Power [dB]")
plt.ylim(0, 4000)
plt.show()

def compare_specs(x, fs, params_default, **kwargs):
    # default parameters
    params = params_default.copy()
    
    # compute default spectrogram
    f, t, Sxx = spectrogram(x, fs,
                            nperseg=params["nperseg"],
                            noverlap=params["noverlap"],
                            window=params["window"])
    
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10*np.log10(Sxx), shading="gouraud")
    plt.title("Default Spectrogram")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.ylim(0, 4000)
    plt.colorbar()
    plt.show()

    # override parameters
    params.update(kwargs)

    f, t, Sxx = spectrogram(x, fs,
                            nperseg=params["nperseg"],
                            noverlap=params["noverlap"],
                            window=params["window"])

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10*np.log10(Sxx), shading="gouraud")
    plt.title("Modified Spectrogram")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.ylim(0, 4000)
    plt.colorbar()
    plt.show()

    #FFT length ↑ -> Frequency resolution ↑, time resolution ↓
    #Window length ↓ -> Time resolution ↑
    #Overlap ↑ -> Smoother plot
    #Hann window -> Good leakage suppression
    # Spectrogram quality strongly depends on FFT size, window length and overlap
    #Speech vowels exhibit strong harmonic and formant structures
    #Consonants are noisy and broadband
    #Hann window offers best balance between leakage and resolution