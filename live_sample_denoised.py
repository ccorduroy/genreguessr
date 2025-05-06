import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import time

# --- Parameters ---
DURATION = 10
FS = 16000
N_FFT = 1024
HOP_LENGTH = N_FFT // 4
WINDOW = signal.hann(N_FFT)

# --- Functions ---
def record_audio(duration, fs):
    print("Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return audio.squeeze()

# STFT
def stft(y, n_fft, hop_length, window):
    return np.array([np.fft.rfft(window * y[i:i+n_fft])
                     for i in range(0, len(y) - n_fft + 1, hop_length)]).T

# inverse STFT
def istft(Zxx, hop_length, window):
    n_fft = len(window)
    time_len = hop_length * (Zxx.shape[1] - 1) + n_fft
    y = np.zeros(time_len)
    W_sum = np.zeros(time_len)

    for i, frame in enumerate(Zxx.T):
        start = i * hop_length
        y[start:start+n_fft] += np.fft.irfft(frame) * window
        W_sum[start:start+n_fft] += window**2

    return y / (W_sum + 1e-10)

# using spectral subtraction denoising technique
def spectral_subtraction(noisy_audio, fs, n_fft, hop_length, window):
    # Estimate noise from first 1 sec
    noise_estimate = noisy_audio[:fs]
    noise_stft = stft(noise_estimate, n_fft, hop_length, window)
    noise_mag = np.abs(noise_stft).mean(axis=1, keepdims=True)

    # Full signal STFT
    noisy_stft = stft(noisy_audio, n_fft, hop_length, window)
    noisy_mag = np.abs(noisy_stft)
    phase = np.angle(noisy_stft)

    # Spectral subtraction
    clean_mag = np.maximum(noisy_mag - noise_mag, 0.0)
    clean_stft = clean_mag * np.exp(1j * phase)

    # Reconstruct time-domain signal
    denoised_audio = istft(clean_stft, hop_length, window)
    return denoised_audio, noisy_mag, noise_mag, clean_mag

def plot_spectrogram(mag, title, fs, hop_length, n_fft):
    plt.figure(figsize=(10, 4))
    plt.imshow(20 * np.log10(mag + 1e-6), aspect='auto', origin='lower', extent=[0, mag.shape[1] * hop_length / fs, 0, fs / 2])
    plt.colorbar(label='dB')
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

# calls
print("Starting recording in")
print("3")
time.sleep(0.5)
print("2")
time.sleep(0.5)
print("1")
time.sleep(0.5)

audio = record_audio(DURATION, FS)
wav.write("raw.wav", FS, (audio * 32767).astype(np.int16))

denoised, noisy_mag, noise_mag, clean_mag = spectral_subtraction(audio, FS, N_FFT, HOP_LENGTH, WINDOW)

# Normalize and save
denoised = denoised / np.max(np.abs(denoised))  # normalize to -1 to 1
wav.write("live_sample.wav", FS, (denoised * 32767).astype(np.int16))
print("Saved 'live_sample.wav'.")

# --- Plot Spectrograms ---
plot_spectrogram(noisy_mag, "Noisy Audio Spectrogram", FS, HOP_LENGTH, N_FFT)
plot_spectrogram(np.tile(noise_mag, (1, noisy_mag.shape[1])), "Estimated Noise Spectrum", FS, HOP_LENGTH, N_FFT)
plot_spectrogram(clean_mag, "Denoised Audio Spectrogram", FS, HOP_LENGTH, N_FFT)
