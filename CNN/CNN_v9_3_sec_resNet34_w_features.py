import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
import cv2
import pickle
import glob
import pandas as pd
from PIL import Image
import os

# PARAMS
# Parameters for the model
n_pixels = 512*512
n_classes = 10

# Parameters for the training
USE_CPU = False
reg_val = 1e-4
lr = 0.001 / 2
batchsize = 32

# For version control
version_number = 9
spectro_len = 3

# LOADING DATA 
class ImageWithUnlinkedFeaturesDataset(Dataset):
    def __init__(self, root_dir, feature_csv, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.features = pd.read_csv(feature_csv).values.astype('float32')

        # Collect image paths in consistent order
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # Get class names from subfolders names
        classes = sorted(os.listdir(root_dir)) 

        for label_idx, class_name in enumerate(classes):
            class_path = os.path.join(root_dir, class_name)
            image_files = sorted(glob.glob(os.path.join(class_path, '*')))
            self.image_paths.extend(image_files)
            self.labels.extend([label_idx] * len(image_files))

            # Optionally store class-to-index mapping
            self.class_to_idx[class_name] = label_idx

    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        # filename = os.path.basename(img_path) # <<< used to verify 

        
        if self.transform:
            image = self.transform(image)

        # Load corresponding features
        feature = torch.tensor(self.features[idx], dtype=torch.float32)

        # Load label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label, feature
    
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[.5], std=[.5])  # Normalize for Grayscale
])

# --- 2. Load dataset ---
dataset = ImageWithUnlinkedFeaturesDataset(
    root_dir=f'sliding_spectrograms_{spectro_len}_seconds',
    feature_csv='features_3_sec.csv',
    transform=transform
)

# Split into train, val, test (70/15/15) since our past models haven't been good with generalization
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size])

#### DO NOT RE-RUN THIS CODE!!!
torch.save(trainset.indices, f'train_indices_v{version_number}.pt')
torch.save(valset.indices, f'val_indices_v{version_number}.pt')
torch.save(testset.indices, f'test_indices_v{version_number}.pt')

image, label, feature = trainset[0] 
print("image shape:",image.shape) # torch.Size([1, 28, 28])
print("label:", label) 
print("feature shape:",feature.shape)
print("features:",feature)
input_image_shape = image.shape
input_feature_shape = feature.shape

print(f'Train set size: {len(trainset)}, Validation set size: {len(valset)}, Test set size: {len(testset)}')



# Shuffle the data at the start of each epoch (only useful for training set)

trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
train_eval_loader = DataLoader(trainset, batch_size=batchsize, shuffle=False)
valloader = DataLoader(valset, batch_size=batchsize, shuffle=False)
testloader = DataLoader(testset, batch_size=batchsize, shuffle=False)


# MODEL - ResNet34
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        super(ResidualBlock,self).__init__()
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self,x):
        residual = x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += self.shortcut(residual)
        x = nn.ReLU(True)(x)
        return x

def classifier_mlp(n_in, n_hidden, n_classes, drate=0.25):
    return nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.ReLU(),
        nn.BatchNorm1d(n_hidden),
        nn.Dropout1d(p=drate),
        nn.Linear(n_hidden, n_classes)
    )

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=2,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(64,64),
            ResidualBlock(64,64,2)
        )
        
        self.block3 = nn.Sequential(
            ResidualBlock(64,128),
            ResidualBlock(128,128,2)
        )
        
        self.block4 = nn.Sequential(
            ResidualBlock(128,256),
            ResidualBlock(256,256,2)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(256,512),
            ResidualBlock(512,512,2)
        )

        self.avgpool = nn.AvgPool2d(2)

        self.classifier = classifier_mlp(8192+57, 1000, 10)
        

    def forward(self,x,features):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        features = features.view(features.size(0),-1)
        x = torch.cat((x,features), dim=1) # Concatenate CNN features with CSV features
        x = self.classifier(x)
        return x

model = ResNet34()
summary(model, [(1,512,512),(1,57,1)])  # call summary before moving the model to a device...
criterion = nn.CrossEntropyLoss() # includes softmax (for numerical stability)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg_val)  # default learning rate is 0.001


# TRAIN/VALIDATE FUNC
# set the device to use and move model to device

if USE_CPU:
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.torch.backends.mps.is_available():
    device = torch.device("mps") # MPS acceleration is available on MacOS 12.3+
else:
    device = torch.device("cpu")

print(f'Using device: {device}')
model.to(device) # Move model to device

# Define function to call for each training epoch (one complete pass over the training set)
def train(model, trainloader, criterion, optimizer, device):
    model.train() # set model to training mode
    running_loss = 0; running_acc = 0
    with tqdm(total=len(trainloader), desc=f"Train", unit="batch") as pbar:
        for n_batch, (images, labels, features) in enumerate(trainloader): # Iterate over batches
            images, labels, features = images.to(device), labels.to(device), features.to(device) # Move batch to device
            # print("train image shape:",images.shape)
            # print("train labels shape:",labels.shape)
            # print("train features shape:",features.shape)
            # print("train feature:", features[0])
            optimizer.zero_grad()
            output = model(images, features) # Forward pass
            loss = criterion(output, labels) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights
            running_loss += loss.item()
            running_acc += (output.argmax(1) == labels).float().mean().item()
            # pbar.set_postfix({'loss': loss.item(), 'acc': 100. * running_acc / (n_batch+1)})
            pbar.set_postfix({'loss': running_loss / (n_batch+1), 'acc': 100. * running_acc / (n_batch+1)})
            pbar.update() # Update progress bar
    return running_loss / len(trainloader), running_acc / len(trainloader) # return loss and accuracy for this epoch

# Define function to call for each validation epoch (one complete pass over the validation set)
def validate(model, valloader, criterion, device, tag='Val'):
    model.eval() # set model to evaluation mode (e.g. turn off dropout, batchnorm, etc.)
    running_loss = 0; running_acc = 0
    with torch.no_grad(): # no need to compute gradients for validation
        with tqdm(total=len(valloader), desc=tag, unit="batch") as pbar:
            for n_batch, (images, labels, features) in enumerate(valloader): # Iterate over batches
                images, labels , features = images.to(device), labels.to(device), features.to(device) # Move batch to device
                output = model(images, features) # Forward pass
                loss = criterion(output, labels) # Compute loss
                running_loss += loss.item() 
                running_acc += (output.argmax(1) == labels).float().mean().item()
                pbar.set_postfix({'loss': running_loss / (n_batch+1), 'acc': 100. * running_acc / (n_batch+1)})
                pbar.update() # Update progress bar
    return running_loss / len(valloader), running_acc / len(valloader)  # return loss and accuracy for this epoch



# TRAINING
# Run training and validation loop
# Save the best model based on validation accuracy
n_epochs = 17
best_acc = -1
train_loss_history = []; train_acc_history = []
val_loss_history = []; val_acc_history = []
for epoch in range(n_epochs): # Iterate over epochs
    print(f"\nEpoch {epoch+1} of {n_epochs}")
    if epoch == n_epochs // 2:
        lr = optimizer.param_groups[0]['lr']
        print(f'Reducing learning rate from {lr} to {lr/4}')
        optimizer.param_groups[0]['lr'] /= 4
    train_loss, train_acc  = train(model, trainloader, criterion, optimizer, device) # Train
    train_loss, train_acc  = validate(model, train_eval_loader, criterion, device, tag='Train Eval') # Evaluate on Train data
    val_loss, val_acc = validate(model, valloader, criterion, device) # Validate
    train_loss_history.append(train_loss); train_acc_history.append(train_acc)
    val_loss_history.append(val_loss); val_acc_history.append(val_acc)
    with open(f'train_loss_history{version_number}.pkl', 'wb') as f:
        pickle.dump(train_loss_history, f)
    with open(f'train_acc_history{version_number}.pkl', 'wb') as f:
        pickle.dump(train_acc_history, f)
    with open(f'val_loss_history{version_number}.pkl', 'wb') as f:
        pickle.dump(val_loss_history, f)
    with open(f'val_acc_history{version_number}.pkl', 'wb') as f:
        pickle.dump(val_acc_history, f)
    if val_acc > best_acc: # Save best model
        best_acc = val_acc
        torch.save(model.state_dict(), f"best_model_v{version_number}.pt") # saving model parameters ("state_dict") saves memory and is faster than saving the entire model

# Saves the last epoch's params
torch.save(model.state_dict(), f"last_model_v{version_number}.pt")

epochs = torch.arange(n_epochs)

# plot training and validation loss
plt.figure()
plt.plot(epochs, train_loss_history, label='train_loss')
plt.plot(epochs, val_loss_history, label='val_loss')
plt.xlabel('epochs')
plt.ylabel('Multiclass Cross Entropy Loss')
plt.title(f'Loss with miniVGG model')
plt.legend()
plt.show()

# plot training and validation accuracy
plt.figure()
plt.plot(epochs, train_acc_history, label='train_acc')
plt.plot(epochs, val_acc_history, label='val_acc')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.title(f'Accuracy with miniVGG model; Regularizer: {reg_val : 3.2g}')
plt.legend()
plt.show()


model.eval() # set model to evaluation mode 


# EVALUATION & ACCURACY
# Load the best model and evaluate on test set
model.load_state_dict(torch.load(f"best_model_v{version_number}.pt"))
test_loss, test_acc = validate(model, testloader, criterion, device)
print(f"Best Model Test accuracy: {test_acc:.4f}")
model.load_state_dict(torch.load(f"last_model_v{version_number}.pt"))
test_loss, test_acc = validate(model, testloader, criterion, device)
print(f"Lastest Model Test accuracy: {test_acc:.4f}")



# CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
import seaborn as sns
 
# model.load_state_dict(torch.load("best_model.pt"))

all_preds = []
all_labels = []
genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

for inputs, labels, features in testloader:  # Replace `testloader` with your DataLoader
        inputs, labels, features = inputs.to(device), labels.to(device), features.to(device)  # Move to the appropriate device (GPU or CPU)
        
        # Forward pass
        outputs = model(inputs, features)
        
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