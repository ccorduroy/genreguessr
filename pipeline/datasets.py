import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
def load_gtzan(path):
    df = pd.read_csv(path)

    df = df.drop(columns=['filename', 'length'])
    features = df.drop(columns=['label'])
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    joblib.dump(scaler, 'scaler.save')

    return features_scaled, df['label']
