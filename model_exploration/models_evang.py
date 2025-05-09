import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes)  
        )
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)

def plot_model_history(history, model_name):
    epochs = range(1, len(history["train_acc"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Trial")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Trial")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_logistic(X_train, y_train, X_val, y_val):
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_acc = 0
    best_model = None

    for C in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(C=C, solver='lbfgs', max_iter=2000)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(1 - train_acc)
        history["val_loss"].append(1 - val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    return best_model, history


def train_svm(X_train, y_train, X_val, y_val):
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_acc = 0
    best_model = None

    for C in [0.01, 0.1, 1, 10, 100]:
        model = SVC(C=C, kernel='linear')
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(1 - train_acc)
        history["val_loss"].append(1 - val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    return best_model, history


def train_perceptron(X_train, y_train, X_val, y_val):
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_acc = 0
    best_model = None

    for eta0 in [0.0001, 0.001, 0.01, 0.1, 1]:
        model = Perceptron(eta0=eta0, max_iter=1000, tol=1e-3)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(1 - train_acc)
        history["val_loss"].append(1 - val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    return best_model, history


def train_ridge(X_train, y_train, X_val, y_val):
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_acc = 0
    best_model = None

    for alpha in [100, 10, 1, 0.1, 0.01]:
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(1 - train_acc)
        history["val_loss"].append(1 - val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    return best_model, history


def train_decision_tree(X_train, y_train, X_val, y_val):
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_acc = 0
    best_model = None

    for max_depth in [1, 3, 5, 10, 20, None]:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(1 - train_acc)
        history["val_loss"].append(1 - val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    return best_model, history


def train_random_forest(X_train, y_train, X_val, y_val):
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_acc = 0
    best_model = None

    for n_estimators in [10, 50, 100, 200]:
        for max_depth in [3, 5, 10, None]:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, model.predict(X_train))
            val_acc = accuracy_score(y_val, model.predict(X_val))
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_loss"].append(1 - train_acc)
            history["val_loss"].append(1 - val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model

    return best_model, history


def train_knn(X_train, y_train, X_val, y_val):
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_acc = 0
    best_model = None

    for n_neighbors in [1, 3, 5, 7, 10]:
        for weights in ['uniform', 'distance']:
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            model.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, model.predict(X_train))
            val_acc = accuracy_score(y_val, model.predict(X_val))
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_loss"].append(1 - train_acc)
            history["val_loss"].append(1 - val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model

    return best_model, history

def train_nn(X_train, y_train, X_val, y_val, n_classes=10, batch_size=32, n_epochs=30, lr=1e-3):
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train_encoded = encoder.transform(y_train)
    y_val_encoded = encoder.transform(y_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    valset = TensorDataset(X_val_tensor, y_val_tensor)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = ImprovedMLP(input_dim, n_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    def train_epoch():
        model.train()
        running_loss = 0
        running_acc = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += (output.argmax(1) == labels).float().mean().item()
        return running_loss / len(trainloader), running_acc / len(trainloader)

    def validate_epoch():
        model.eval()
        running_loss = 0
        running_acc = 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                running_loss += loss.item()
                running_acc += (output.argmax(1) == labels).float().mean().item()
        return running_loss / len(valloader), running_acc / len(valloader)

    best_acc = -1
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch()
        val_loss, val_acc = validate_epoch()
        scheduler.step(val_acc)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)

    class ModelWrapper:
        def __init__(self, model, device, encoder):
            self.model = model
            self.device = device
            self.encoder = encoder

        def predict(self, X):
            self.model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                _, preds = torch.max(outputs, 1)
            return self.encoder.inverse_transform(preds.cpu().numpy())

    return ModelWrapper(model, device, encoder), history
