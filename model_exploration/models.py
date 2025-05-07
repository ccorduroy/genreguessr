from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score


def train_logistic(X_train, y_train, X_val, y_val):
    best_acc = 0
    best_C = None
    for C in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(C=C, solver='lbfgs', max_iter=2000)
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        if val_acc > best_acc:
            best_acc = val_acc
            best_C = C
    final_model = LogisticRegression(C=best_C, solver='lbfgs', max_iter=2000)
    final_model.fit(X_train, y_train)
    return final_model

def train_svm(X_train, y_train, X_val, y_val):
    best_acc = 0
    best_C = None
    for C in [0.01, 0.1, 1, 10, 100]:
        model = SVC(C=C, kernel='linear') 
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        if val_acc > best_acc:
            best_acc = val_acc
            best_C = C
    final_model = SVC(C=best_C, kernel='linear')
    final_model.fit(X_train, y_train)
    return final_model

def train_perceptron(X_train, y_train, X_val, y_val):
    best_acc = 0
    best_eta0 = None
    for eta0 in [0.0001, 0.001, 0.01, 0.1, 1]:
        model = Perceptron(eta0=eta0, max_iter=1000, tol=1e-3)
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        if val_acc > best_acc:
            best_acc = val_acc
            best_eta0 = eta0
    final_model = Perceptron(eta0=best_eta0, max_iter=1000, tol=1e-3)
    final_model.fit(X_train, y_train)
    return final_model

def train_ridge(X_train, y_train, X_val, y_val):
    best_acc = 0
    best_alpha = None
    for alpha in [100, 10, 1, 0.1, 0.01]: 
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        if val_acc > best_acc:
            best_acc = val_acc
            best_alpha = alpha
    final_model = RidgeClassifier(alpha=best_alpha)
    final_model.fit(X_train, y_train)
    return final_model

def train_decision_tree(X_train, y_train, X_val, y_val):
    best_acc = 0
    best_depth = None
    for max_depth in [1, 3, 5, 10, 20, None]:  
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        if val_acc > best_acc:
            best_acc = val_acc
            best_depth = max_depth
    final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    final_model.fit(X_train, y_train)
    return final_model

def train_random_forest(X_train, y_train, X_val, y_val):
    best_acc = 0
    best_params = None
    for n_estimators in [10, 50, 100, 200]:
        for max_depth in [3, 5, 10, None]:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            val_acc = accuracy_score(y_val, model.predict(X_val))
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = (n_estimators, max_depth)
    final_model = RandomForestClassifier(n_estimators=best_params[0], max_depth=best_params[1], random_state=42)
    final_model.fit(X_train, y_train)
    return final_model

def train_knn(X_train, y_train, X_val, y_val):
    best_acc = 0
    best_params = None
    for n_neighbors in [1, 3, 5, 7, 10]:  
        for weights in ['uniform', 'distance']:  
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            model.fit(X_train, y_train)
            val_acc = accuracy_score(y_val, model.predict(X_val))
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = (n_neighbors, weights)
    
    final_model = KNeighborsClassifier(n_neighbors=best_params[0], weights=best_params[1])
    final_model.fit(X_train, y_train)
    return final_model

# --------------------------------------------#
# NEURAL NET CODE #
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm

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

def train_nn(X_train, y_train, X_val, y_val, n_classes=10, batch_size=32, n_epochs=30, lr=1e-3):
    # Setup device
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    def train_epoch(model, trainloader, criterion, optimizer, device):
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
    
    def validate_epoch(model, valloader, criterion, device):
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
        # print(f"Epoch {epoch+1}/{n_epochs}")
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, valloader, criterion, device)
        scheduler.step(val_acc)

        # print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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
        
        def predict_proba(self, X):
            self.model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()
    
    return ModelWrapper(model, device, encoder)