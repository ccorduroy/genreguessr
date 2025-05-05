import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr

DURATION = 10
FS = 44100  # wav sample rate = 44.1 kHz


def record_audio(duration, fs):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return audio.flatten()


def save_wav(filename, audio_data, fs):
    wav.write(filename, fs, (audio_data * 32767).astype(np.int16))  # normalize to 16-bit PCM
    print(f"Saved denoised audio to {filename}")


def main():
    raw_audio = record_audio(DURATION, FS)

    # Estimate noise using the first second
    noise_profile = raw_audio[:FS]  # first second as noise sample
    denoised_audio = nr.reduce_noise(y=raw_audio, sr=FS, y_noise=noise_profile, prop_decrease=1.0)

    save_wav("recording.wav", denoised_audio, FS)


if __name__ == "__main__":
    main()
