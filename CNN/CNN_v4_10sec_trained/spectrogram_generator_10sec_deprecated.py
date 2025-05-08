import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from PIL import Image


def visualize_single_spectrogram(file_name):
    # Load audio file
    y, sr = librosa.load(file_name)

    # Compute spectrogram (Short-Time Fourier Transform)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(D), ref=np.max)

    # Display the spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')  # y_axis='linear' or 'log'
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()
    plt.close()


def create_single_spectrogram(file_path, duration):
    y, sr = librosa.load(file_path)
    # print("sampling rate: ", sr)
    total_duration = librosa.get_duration(y=y, sr=sr)
    adjusted_length = int(duration * sr)
    adjusted_y = y[0:adjusted_length]
    # Create spectrogram
    # Perform Short-time Fourier transform (STFT)
    D = librosa.stft(adjusted_y)  # np.abs(D[..., f, t]) is the magnitude of frequency bin f at frame t
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Converts amplitude from V to dB

    S_db_normalized = cv2.normalize(S_db, None, 0, 255, cv2.NORM_MINMAX)
    S_db_normalized = np.uint8(S_db_normalized)

    # Create colored maps & Resize/Rotate
    color_map = cv2.COLORMAP_INFERNO  # Looks similar to matplotlib's 'magma'
    colored_img = cv2.applyColorMap(S_db_normalized, color_map)
    resized_img = cv2.resize(colored_img, (640, 480), interpolation=cv2.INTER_AREA)
    rotated_image = cv2.rotate(resized_img, cv2.ROTATE_180)

    output_path = f"{os.path.splitext(file_path)[0]}_spectrogram.png"

    cv2.imwrite(output_path, rotated_image)


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
                y, sr = librosa.load(input_path)
                # print("sampling rate: ", sr)
                total_duration = librosa.get_duration(y=y, sr=sr)

                # Convert time-based window/overlap to samples
                window_length = int(window_duration * sr)
                overlap_length = int(overlap_duration * sr)

                for i, start_sample in enumerate(range(0, len(y) - window_length + 1, overlap_length)):
                    end_sample = start_sample + window_length
                    y_chunk = y[start_sample:end_sample]

                    # Create spectrogram
                    # Perform Short-time Fourier transform (STFT)
                    D = librosa.stft(y_chunk)  # np.abs(D[..., f, t]) is the magnitude of frequency bin f at frame t
                    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Converts amplitude from V to dB

                    S_db_normalized = cv2.normalize(S_db, None, 0, 255, cv2.NORM_MINMAX)
                    S_db_normalized = np.uint8(S_db_normalized)

                    # Create colored maps & Resize/Rotate
                    # color_map = cv2.COLORMAP_INFERNO  # Looks similar to matplotlib's 'magma'
                    colored_img = S_db_normalized  # cv2.applyColorMap(S_db_normalized, color_map)
                    resized_img = cv2.resize(colored_img, (640, 480), interpolation=cv2.INTER_AREA)
                    rotated_image = cv2.rotate(resized_img, cv2.ROTATE_180)

                    output_name = f"{os.path.splitext(file)[0]}_win{i}.png"
                    output_path = os.path.join(temp_output_folder, output_name)

                    cv2.imwrite(output_path, rotated_image)

                    # Plot and save
                    # plt.figure(figsize=(10, 4))     # FOR GRAPH VISUALS
                    # librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')     # FOR GRAPH VISUALS
                    # plt.colorbar(format='%+2.0f dB')     # FOR GRAPH VISUALS
                    # plt.title(f"{file} - window {i}")     # FOR GRAPH VISUALS
                    # plt.tight_layout()
                    # plt.savefig(output_path)
                    # plt.close()

                    print(f"Saved: {output_path}")


# -------------------------------------------------------------------------

# To generate only 1 image

#filePath = "blues.00000.wav"
#create_single_spectrogram(file_path=filePath, duration=30)

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

# CAUTION: this will eat up your memory so don't run it on a big folder
create_spectrogram_sliding_window(window_duration=10, overlap_duration=5, parent_folder='./gtzan/genres_original')

# -------------------------------------------------------------------------

# To check what image type is (and therefore # of channels, aka depth):
# Mode "L" means grayscale (1 channel).
# Mode "RGB" means color image (3 channels).
# Mode "RGBA" means color image with transparency (4 channels).

#img = Image.open("sliding_spectrograms_10_seconds/rock/rock.00094_win0.png")
#width, height = img.size
#print(img.mode)
#print(f"Width: {width}, Height: {height}")