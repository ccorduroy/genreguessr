# GPU enabled (cuda)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

#print(torch.__version__)
# set device (used globally)
device = (
    torch.device("cuda" if torch.cuda.is_available() else
                 "mps" if torch.backends.mps.is_available() else
                 "cpu")
)

print(f'Using device: {device}')

# Parameters for the model
n_pixels = 28 * 28
n_classes = 10

# Parameters for the training
reg_val = 1e-4
lr = 0.001 / 2
batchsize = 32

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts [0-255] RGBA to [0.0-1.0], shape [3, H, W]
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # Normalize for RGB
])

# --- 2. Load dataset ---
dataset = datasets.ImageFolder(root='sliding_spectrograms_10_seconds', transform=transform)

# Split into train, val, test (80/10/10)
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size])

torch.save(trainset.indices, 'train_indices.pt')
torch.save(valset.indices, 'val_indices.pt')
torch.save(testset.indices, 'test_indices.pt')

image, label = trainset[0]
print(image.shape)  # torch.Size([1, 28, 28])
print(label)
input_image_shape = image.shape

print(f'Train set size: {len(trainset)}, Validation set size: {len(valset)}, Test set size: {len(testset)}')

class FullDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data, self.labels = self._load_data()

    def _load_data(self):
        data = []
        labels = []

        for i in range(len(self.dataset)):
            x, y = self.dataset[i]
            data.append(x)
            labels.append(y)

        return torch.stack(data), torch.tensor(labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.data[idx].to(device), self.labels[idx].to(device)


trainset = FullDataset(trainset)
valset = FullDataset(valset)

# Shuffle the data at the start of each epoch (only useful for training set)

trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
train_eval_loader = DataLoader(trainset, batch_size=batchsize, shuffle=False)
valloader = DataLoader(valset, batch_size=batchsize, shuffle=False)
testloader = DataLoader(testset, batch_size=batchsize, shuffle=False)


# Define the model
# for an alternative approach, see: mini_vgg_like_torch_1.py
def vgg_like_block(in_channels, num_filters, ksize=3, drate=0.25, pad='same'):
    return nn.Sequential(
        nn.Conv2d(in_channels, num_filters, kernel_size=ksize, padding=pad),
        nn.ReLU(),
        nn.BatchNorm2d(num_filters),
        nn.Conv2d(num_filters, num_filters, kernel_size=ksize, padding=pad),
        nn.ReLU(),
        nn.BatchNorm2d(num_filters),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout2d(p=drate)
    )


def classifier_mlp(n_in, n_hidden, n_classes, drate=0.25):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_in, n_hidden),
        nn.ReLU(),
        nn.BatchNorm1d(n_hidden),
        nn.Dropout1d(p=drate),
        nn.Linear(n_hidden, n_classes)
    )


class MiniVgg(nn.Module):
    def __init__(self, b1_filters=32, b2_filters=64, H=480, W=640, fc_nodes=512, n_classes=10):
        super().__init__()
        self.block1 = vgg_like_block(3, b1_filters)
        self.block2 = vgg_like_block(b1_filters, b2_filters)
        assert H % 4 == 0, f'the image height and width must be a multiple of 4: you passed H = {H}'
        mlp_in_size = (H * W // 16) * b2_filters  # the H and W are both reduced by 4 with 2 max-pool layers.
        self.classifier = classifier_mlp(mlp_in_size, fc_nodes, n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        y = self.block1(x)
        y = self.block2(y)
        y = y.view(batch_size, -1)
        y = self.classifier(y)
        return y

# setup
model = MiniVgg()
# set the device to use and move model to device
model.to(device)  # Move model to device
print(model)
summary(model, input_image_shape)  # call summary before moving the model to a device...


criterion = nn.CrossEntropyLoss()  # includes softmax (for numerical stability)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg_val)  # default learning rate is 0.001

# Define function to call for each training epoch (one complete pass over the training set)
def train(model, trainloader, criterion, optimizer, device):
    model.train()  # set model to training mode
    running_loss = 0;
    running_acc = 0
    with tqdm(total=len(trainloader), desc=f"Train", unit="batch") as pbar:
        for n_batch, (images, labels) in enumerate(trainloader):  # Iterate over batches
            images, labels = images.to(device), labels.to(device)  # Move batch to device
            optimizer.zero_grad()
            output = model(images)  # Forward pass
            loss = criterion(output, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item()
            running_acc += (output.argmax(1) == labels).float().mean().item()
            # pbar.set_postfix({'loss': loss.item(), 'acc': 100. * running_acc / (n_batch+1)})
            pbar.set_postfix({'loss': running_loss / (n_batch + 1), 'acc': 100. * running_acc / (n_batch + 1)})
            pbar.update()  # Update progress bar
    return running_loss / len(trainloader), running_acc / len(trainloader)  # return loss and accuracy for this epoch


# Define function to call for each validation epoch (one complete pass over the validation set)
def validate(model, valloader, criterion, device, tag='Val'):
    model.eval()  # set model to evaluation mode (e.g. turn off dropout, batchnorm, etc.)
    running_loss = 0;
    running_acc = 0
    with torch.no_grad():  # no need to compute gradients for validation
        with tqdm(total=len(valloader), desc=tag, unit="batch") as pbar:
            for n_batch, (images, labels) in enumerate(valloader):  # Iterate over batches
                images, labels = images.to(device), labels.to(device)  # Move batch to device
                output = model(images)  # Forward pass
                loss = criterion(output, labels)  # Compute loss
                running_loss += loss.item()
                running_acc += (output.argmax(1) == labels).float().mean().item()
                pbar.set_postfix({'loss': running_loss / (n_batch + 1), 'acc': 100. * running_acc / (n_batch + 1)})
                pbar.update()  # Update progress bar
    return running_loss / len(valloader), running_acc / len(valloader)  # return loss and accuracy for this epoch


# Run training and validation loop
# Save the best model based on validation accuracy
n_epochs = 20
best_acc = -1
train_loss_history = [];
train_acc_history = []
val_loss_history = [];
val_acc_history = []
#for epoch in range(n_epochs):  # Iterate over epochs
#    print(f"\nEpoch {epoch + 1} of {n_epochs}")
#    if epoch == n_epochs // 2:
#        lr = optimizer.param_groups[0]['lr']
#        print(f'Reducing learning rate from {lr} to {lr / 4}')
#        optimizer.param_groups[0]['lr'] /= 4
#    train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)  # Train
#    train_loss, train_acc = validate(model, train_eval_loader, criterion, device,
#                                     tag='Train Eval')  # Evaluate on Train data
#    val_loss, val_acc = validate(model, valloader, criterion, device)  # Validate
#    train_loss_history.append(train_loss);
#    train_acc_history.append(train_acc)
#    val_loss_history.append(val_loss);
#    val_acc_history.append(val_acc)
#    if val_acc > best_acc:  # Save best model
#        best_acc = val_acc
#        torch.save(model.state_dict(),
#                   "best_model_v4.pt")  # saving model parameters ("state_dict") saves memory and is faster than saving the entire model
#
#epochs = torch.arange(n_epochs)

# plot training and validation loss
#plt.figure()
#plt.plot(epochs, train_loss_history, label='train_loss')
#plt.plot(epochs, val_loss_history, label='val_loss')
#plt.xlabel('epochs')
#plt.ylabel('Multiclass Cross Entropy Loss')
#plt.title(f'Loss with miniVGG model')
#plt.legend()
#plt.show()

# plot training and validation accuracy
#plt.figure()
#plt.plot(epochs, train_acc_history, label='train_acc')
#plt.plot(epochs, val_acc_history, label='val_acc')
#plt.xlabel('epochs')
#plt.ylabel('Accuracy')
#plt.title(f'Accuracy with miniVGG model; Regularizer: {reg_val : 3.2g}')
#plt.legend()
#plt.show()

# Load the best model and evaluate on test set
model.load_state_dict(torch.load("best_model_v4.pt"))
test_loss, test_acc = validate(model, testloader, criterion, device)
print(f"Test accuracy: {test_acc:.4f}")

model.eval()  # set model to evaluation mode

# model.load_state_dict(torch.load("best_model.pt"))

all_preds = []
all_labels = []
genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

for inputs, labels in testloader:  # Replace `testloader` with your DataLoader
    inputs, labels = inputs.to(device), labels.to(device)  # Move to the appropriate device (GPU or CPU)

    # Forward pass
    outputs = model(inputs)

    # Get the predictions
    _, preds = torch.max(outputs, 1)

    # Store all predictions and labels
    all_preds.extend(preds.cpu().numpy())  # Convert to numpy and add to list
    all_labels.extend(labels.cpu().numpy())  # Convert to numpy and add to list

# Create the confusion matrix
cm = confusion_matrix(all_labels, all_preds)
class_labels = [genre_list[label] for label in all_labels]

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_list, yticklabels=genre_list)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# --------------------------------------------

# Recreating Test Set

# Rebuild the original dataset
dataset = datasets.ImageFolder(root='sliding_spectrograms_10_seconds', transform=transform)

# Load indices
train_indices = torch.load('train_indices.pt')
val_indices = torch.load('val_indices.pt')
test_indices = torch.load('test_indices.pt')

# Recreate test subset
trainset = Subset(dataset, train_indices)
valset = Subset(dataset, val_indices)
testset = Subset(dataset, test_indices)

batchsize = 32

# Create DataLoader

# trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
# train_eval_loader = DataLoader(trainset, batch_size=batchsize, shuffle=False)
# valloader = DataLoader(valset, batch_size=batchsize, shuffle=False)
testloader = DataLoader(testset, batch_size=batchsize, shuffle=False)