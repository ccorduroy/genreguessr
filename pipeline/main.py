from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import load_gtzan
from transformations import no_transform, pca_transform, rbf_transform, polynomial_kernel

from models import train_logistic, train_svm, train_perceptron, train_ridge, train_decision_tree, train_random_forest, train_knn, train_nn
import pandas as pd

import joblib
import os
import torch
import inspect

os.makedirs("saved_models", exist_ok=True)

transformations = {
    "none": no_transform,
    "pca": pca_transform,
    "rbf_kernel" : rbf_transform,
    "polynomial_kernel" : polynomial_kernel
}

models = {
    "logistic_regression": train_logistic,
    "svm": train_svm,
    "perceptron": train_perceptron,
    "ridge": train_ridge,
    "decision_tree": train_decision_tree,
    "random_forest": train_random_forest,
    "KNN": train_knn,
    "neural_net" : train_nn
}

X, y = load_gtzan('../GTZAN_Dataset/features_3_sec.csv')

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


results_dict = {}
for tf_name, tf_func in transformations.items():
    X_train_transformed, X_val_transformed, X_test_transformed = tf_func(X_train, X_val, X_test)

    results_dict[tf_name] = {}

    for model_name, model_func in models.items():
        model = model_func(X_train_transformed, y_train, X_val_transformed, y_val)
        acc = accuracy_score(y_test, model.predict(X_test_transformed))
        
        results_dict[tf_name][model_name] = round(acc, 4)

        print(f"[{tf_name}] + [{model_name}] = Test Accuracy: {acc:.4f}")

        if model_name == "neural_net":
            filename = f"saved_models/{tf_name}__{model_name}.pt"
            # torch.save(model.state_dict(), filename)
        else:
            filename = f"saved_models/{tf_name}__{model_name}.joblib"
            joblib.dump(model, filename)

results_df = pd.DataFrame(results_dict)
print("\nFinal Results Table:")
print(results_df)
