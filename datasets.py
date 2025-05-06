import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_gtzan(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['filename'])
    features = df.drop(columns=['label'])
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, df['label']
