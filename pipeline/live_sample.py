import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import time
from inference import audio_prediction
# --- Parameters ---
FS = 16000
NOISE_DURATION = 1
MUSIC_DURATION = 10
N_FFT = 1024
HOP_LENGTH = N_FFT // 4
WINDOW = signal.windows.hann(N_FFT)

def record_audio(duration, fs, label="audio"):
    print(f"Listening for {label} ({duration} seconds)...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print(f"{label.capitalize()} recording complete.")
    return audio.squeeze()

def stft(y, n_fft, hop_length, window):
    return np.array([np.fft.rfft(window * y[i:i+n_fft])
                     for i in range(0, len(y) - n_fft + 1, hop_length)]).T

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

# spectral subtraction
# from separate noise-only sample (1 sec) and goal audio sample (10 sec)
# takes the STFT of the noise and the music and their magnitudes
# cleans by subtracting the noise from the music by magnitude and phase shifting
# perform inverse STFT on this phased subtraction to reconstruct the time-domain signal

def spectral_subtraction(noise_audio, noisy_audio, fs, n_fft, hop_length, window):
    # Estimate noise spectrum
    noise_stft = stft(noise_audio, n_fft, hop_length, window)
    noise_mag = np.abs(noise_stft).mean(axis=1, keepdims=True)

    # STFT of full signal
    noisy_stft = stft(noisy_audio, n_fft, hop_length, window)
    noisy_mag = np.abs(noisy_stft)
    phase = np.angle(noisy_stft)

    # Spectral subtraction
    clean_mag = np.maximum(noisy_mag - noise_mag, 0.0)
    clean_stft = clean_mag * np.exp(1j * phase)

    # Reconstruct time-domain signal
    denoised_audio = istft(clean_stft, hop_length, window)
    return denoised_audio, noisy_mag, noise_mag, clean_mag

# plotting just for demonstration
def plot_spectrogram(mag, title, fs, hop_length, n_fft):
    plt.figure(figsize=(10, 4))
    plt.imshow(20 * np.log10(mag + 1e-6), aspect='auto', origin='lower',
               extent=[0, mag.shape[1] * hop_length / fs, 0, fs / 2])
    plt.colorbar(label='dB')
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

# --- Main Workflow ---
noise_audio = record_audio(NOISE_DURATION, FS, label="background noise sample")

print("Recording music, get ready:")
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)

music_audio = record_audio(MUSIC_DURATION, FS, label="music sample")
wav.write("raw.wav", FS, (music_audio * 32767).astype(np.int16))

denoised, noisy_mag, noise_mag, clean_mag = spectral_subtraction(noise_audio, music_audio, FS, N_FFT, HOP_LENGTH, WINDOW)

# Normalize and save
denoised = denoised / np.max(np.abs(denoised))
wav.write("live_sample.wav", FS, (denoised * 32767).astype(np.int16))
print("\nSaved denoised file as 'live_sample.wav'.")

audio_prediction("live_sample.wav")

