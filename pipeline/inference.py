import joblib 
from datasets import load_gtzan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import librosa
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def extract_features_by_segments(file_path,  num_segments=10, sample_rate=22050, num_mfcc=20, n_fft=2048, hop_length=512):
    try:
        y, sr = librosa.load(file_path, sr=sample_rate, duration=30.0)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    samples_per_segment = int(sample_rate * 30 / num_segments)
    segment_features = []

    for n in range(num_segments):
        start_sample = n * samples_per_segment
        end_sample = start_sample + samples_per_segment
        y_seg = y[start_sample:end_sample]

        if len(y_seg) < samples_per_segment:
            # Pad if the segment is too short
            y_seg = np.pad(y_seg, (0, samples_per_segment - len(y_seg)))

        features = []

        # Chromagram
        chromagram = librosa.feature.chroma_stft(y=y_seg, sr=sample_rate, hop_length=hop_length)
        features.append(chromagram.mean())
        features.append(chromagram.var())

        # RMS
        rms = librosa.feature.rms(y=y_seg)
        features.append(rms.mean())
        features.append(rms.var())

        # Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y_seg, sr=sample_rate)
        features.append(spec_cent.mean())
        features.append(spec_cent.var())

        # Spectral Bandwidth
        spec_band = librosa.feature.spectral_bandwidth(y=y_seg, sr=sample_rate)
        features.append(spec_band.mean())
        features.append(spec_band.var())

        # Spectral Rolloff
        spec_roll = librosa.feature.spectral_rolloff(y=y_seg, sr=sample_rate)
        features.append(spec_roll.mean())
        features.append(spec_roll.var())

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y_seg)
        features.append(zcr.mean())
        features.append(zcr.var())

        # Harmonics and Percussive
        harmony, perceptr = librosa.effects.hpss(y=y_seg)
        features.append(harmony.mean())
        features.append(harmony.var())
        features.append(perceptr.mean())
        features.append(perceptr.var())

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y_seg, sr=sample_rate)
        features.append(tempo[0])

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y_seg, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        for i in range(num_mfcc):
            features.append(mfcc[:, i].mean())
            features.append(mfcc[:, i].var())

        segment_features.append(np.array(features, dtype=np.float64))

    return np.array(segment_features) 


filename = "saved_models/none__KNN.joblib"
model = joblib.load(filename)

X, y = load_gtzan('../GTZAN_Dataset/features_30_sec.csv')

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


def audio_prediction(path):

    data_modified = extract_features_by_segments(path, 1)
    data_modified_df = pd.DataFrame(data_modified)

    scaler = joblib.load('scaler.save')
    df_scaled = pd.DataFrame(scaler.transform(data_modified_df))

    data_modified = scaler.transform(data_modified)

    print(model.predict(data_modified))

# audio_prediction("live_sample.wav")
# audio_prediction("../GTZAN_Dataset/genres_original/blues/blues.00000.wav")

