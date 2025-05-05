# takes a 10s sample of audio and uses standard signal processing techniques to denoise.
# really records for 11 seconds but the first second is used to estimate noise.

import numpy as np
import sounddevice as sd
import scipy.signal
import scipy.io.wavfile as wav
import time
import matplotlib.pyplot as plt

#globals
FS = 44100  # standard wav encoding rate
NOISE_DURATION = 1
MUSIC_DURATION = 10
N_FFT = 1024
HOP_LENGTH = 512

def record_audio(duration, FS):
    print(f"Recording {duration} seconds...")
    audio = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

# Weiner filter implementation:
# uses STFT to analyze the signal in time-frequency space
# uses the Weiner gain function:
#       G(f, t) = S(f, t)^2 / (S(f, t)^2 + N(f, t)^2)
# inverse STFT reconstructs the cleaned signal

def wiener_denoise(signal, noise_sample):
    # compute STFTs
    _, _, Sxx = scipy.signal.stft(signal, FS, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    _, _, Nxx = scipy.signal.stft(noise_sample, FS, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)

    # power spectra
    S_power = np.abs(Sxx) ** 2
    N_power = np.mean(np.abs(Nxx) ** 2, axis=1, keepdims=True)

    # wiener gain function
    G = S_power / (S_power + N_power + 1e-10)  # avoid divide by zero
    S_denoised = G * Sxx

    # inverse STFT
    _, denoised_signal = scipy.signal.istft(S_denoised, FS, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)

    return denoised_signal, Sxx, Nxx, S_denoised    # last 3 are for graphing purposes later

def save_audio(filename, audio, FS):
    audio = audio / np.max(np.abs(audio))  # normalize to [-1, 1]
    audio_int16 = np.int16(audio * 32767)
    wav.write(filename, FS, audio_int16)
    print(f"Saved to {filename}")

def db(x): return 20 * np.log10(np.abs(x) + 1e-6)   # decibel conversion


if __name__ == "__main__":
    print("WAIT --- Recording 1 second of background noise...")
    noise_sample = record_audio(NOISE_DURATION, FS)

    print("GET READY --- Recording 10 seconds of music...")
    print("3...")
    time.sleep(0.5)
    print("2...")
    time.sleep(0.5)
    print("1...")
    time.sleep(0.5)
    print("Listening...")
    music_signal = record_audio(MUSIC_DURATION, FS)

    # uses signal and noise sample from separate recordings
    denoised, Sxx_signal, Sxx_noise, Sxx_denoised = wiener_denoise(music_signal, noise_sample)
    # save as .wav
    save_audio("wiener_denoised_music.wav", denoised, FS)

    # graphing spectrograms before and after for reference
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axs[0].imshow(db(Sxx_noise), aspect='auto', origin='lower', extent=[0, Sxx_noise.shape[1] * HOP_LENGTH / FS, 0, FS / 2])
    axs[0].set_title("Noise-Only STFT")
    axs[0].set_ylabel("Frequency [Hz]")

    axs[1].imshow(db(Sxx_signal), aspect='auto', origin='lower', extent=[0, Sxx_signal.shape[1] * HOP_LENGTH / FS, 0, FS / 2])
    axs[1].set_title("Noisy Music STFT")
    axs[1].set_ylabel("Frequency [Hz]")

    axs[2].imshow(db(Sxx_denoised), aspect='auto', origin='lower', extent=[0, Sxx_denoised.shape[1] * HOP_LENGTH / FS, 0, FS / 2])
    axs[2].set_title("Denoised Music STFT (Wiener Filtered)")
    axs[2].set_xlabel("Time [sec]")
    axs[2].set_ylabel("Frequency [Hz]")

    plt.tight_layout()
    plt.show()