from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures

def no_transform(X_train, X_val, X_test):
    return X_train.copy(), X_val.copy(), X_test.copy()

def pca_transform(X_train, X_val, X_test, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold)
    pca.fit(X_train)
    return pca.transform(X_train), pca.transform(X_val), pca.transform(X_test)

def rbf_transform(X_train, X_val, X_test):
    beta = 0.025
    k = 200

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    heads = kmeans.cluster_centers_

    def transformation(x_input, cluster_heads, beta):
        squared_distances = cdist(x_input, cluster_heads, metric='sqeuclidean')
        return np.exp(-1 * beta * squared_distances)

    X_train_transformed = transformation(X_train, heads, beta)
    X_val_transformed = transformation(X_val, heads, beta)
    X_test_transformed = transformation(X_test, heads, beta)

    return X_train_transformed, X_val_transformed, X_test_transformed

def polynomial_kernel(X_train, X_val, X_test):

    degree = 2
    poly = PolynomialFeatures(degree)

    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)

    return X_train_poly, X_val_poly, X_test_poly

