# currently set for 10 3-second segments

import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from PIL import Image

N_FFT = 1024
HOP_LENGTH = 71     # sample_rate * time / n_fft to make the spectrogram relatively square
SAMPLE_RATE = 12000
HEIGHT = N_FFT//2


def generate_spectrogram(y):
    # Compute spectrogram (Short-Time Fourier Transform)
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Only keep the first 512 bins (due to N/2 symmetry)
    S_db = S_db[:512, :]
    return S_db


def visualize_single_spectrogram(file_name):
    # Load audio file and downsample
    y, sr = librosa.load(file_name, sr=SAMPLE_RATE)  # Downsample to 12kHz

    # Generate spectrogram
    S_db = generate_spectrogram(y)

    S_db = S_db[:HEIGHT, :]

    # Display the spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')  # y_axis='log'
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()
    plt.close()


def create_single_spectrogram(file_path, duration):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)  # Downsample to 12kHz
    total_duration = librosa.get_duration(y=y, sr=sr)
    adjusted_length = int(duration * sr)
    adjusted_y = y[0:adjusted_length]

    # Generate spectrogram
    S_db = generate_spectrogram(adjusted_y)

    S_db = S_db[:HEIGHT, :]

    # Normalize and convert to uint8 for image format
    S_db_normalized = cv2.normalize(S_db, None, 0, 255, cv2.NORM_MINMAX)
    S_db_normalized = np.uint8(S_db_normalized)

    # Save the spectrogram image (greyscale)
    output_path = f"{os.path.splitext(file_path)[0]}_spectrogram.png"
    cv2.imwrite(output_path, S_db_normalized)


def create_spectrogram_sliding_window(window_duration, overlap_duration, parent_folder, genre=None):
    # Settings
    if (genre is None):
        input_folder = f'{parent_folder}'
    else:
        input_folder = f'{parent_folder}/{genre}'
    output_folder = f'sliding_spectrograms_{window_duration}_seconds'

    # Walk through WAV files
    for root, dirs, files in os.walk(input_folder):
        # print (f"root: {root}   dirs: {dirs}  files: {files}")     #FOR DEBUGGING
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                # relative_path = os.path.relpath(root, input_folder) # ONLY USED FOR "genres_original"
                relative_path = os.path.basename(root)
                temp_output_folder = os.path.join(output_folder, relative_path)
                # print("input path: ", input_path)    #FOR DEBUGGING
                # print("relative path: ", relative_path)    #FOR DEBUGGING
                # print("output path: ", temp_output_folder)    #FOR DEBUGGING

                os.makedirs(temp_output_folder, exist_ok=True)

                # Load audio
                y, sr = librosa.load(input_path, sr=SAMPLE_RATE)
                #print("sampling rate: ", sr)
                total_duration = librosa.get_duration(y=y, sr=sr)

                # Convert time-based window/overlap to samples
                window_length = int(window_duration * sr)
                overlap_length = int(overlap_duration * sr)

                for i, start_sample in enumerate(range(0, len(y) - window_length + 1, overlap_length)):
                    end_sample = start_sample + window_length
                    y_chunk = y[start_sample:end_sample]

                    # Create spectrogram
                    # Perform Short-time Fourier transform (STFT)
                    D = librosa.stft(y_chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)  # np.abs(D[..., f, t]) is the magnitude of frequency bin f at frame t
                    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Converts amplitude from V to dB

                    # take half of bins due to N/2 symmetry
                    S_db = S_db[:HEIGHT, :]

                    S_db_normalized = cv2.normalize(S_db, None, 0, 255, cv2.NORM_MINMAX)

                    # add border to make width 512 (is 508 with current 3-second implementation)
                    S_db_padded = cv2.copyMakeBorder(S_db_normalized, 0, 0, 2, 2, cv2.BORDER_CONSTANT, value=0)

                    S_db_padded = np.uint8(S_db_padded)

                    output_name = f"{os.path.splitext(file)[0]}_win{i}.png"
                    output_path = os.path.join(temp_output_folder, output_name)

                    # keep BW (no colormap)
                    cv2.imwrite(output_path, S_db_padded)

                    # Plot and save
                    # plt.figure(figsize=(10, 4))     # FOR GRAPH VISUALS
                    # librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')     # FOR GRAPH VISUALS
                    # plt.colorbar(format='%+2.0f dB')     # FOR GRAPH VISUALS
                    # plt.title(f"{file} - window {i}")     # FOR GRAPH VISUALS
                    # plt.tight_layout()
                    # plt.savefig(output_path)
                    # plt.close()

                    print(f"Saved: {output_path}")


# %%
# To generate only 1 image
# filePath = "blues.00000.wav"
# create_single_spectrogram(file_path=filePath, duration=30)

# -------------------------------------------------------------------------

#  To generate spectrograms for only 1 genre

# create_spectrogram_sliding_window(window_duration=10, overlap_duration=5, parent_folder='genres_original', genre="rock")

# -------------------------------------------------------------------------

# To generate whatever genre is in the list

# genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
# for genre in genre_list:
#     create_spectrogram_sliding_window(window_duration=10, overlap_duration=5, parent_folder='genres_original' genre=genre)

# -------------------------------------------------------------------------

# To generate spectrogram for all genre in the parent folder

create_spectrogram_sliding_window(window_duration=3, overlap_duration=3, parent_folder='./gtzan/genres_original')
# create_spectrogram_sliding_window(window_duration=10, overlap_duration=10, parent_folder='genres_original',genre="blues")