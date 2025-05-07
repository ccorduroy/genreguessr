import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

import pickle

# Parameters for the model
n_pixels = 512*512
n_classes = 10

# For version control
version_number = 6
spectro_len = 3     # use 3-second spectrogram set

# --- RECREATE TEST SET AFTER TRAINING FIRST TIME ---

# Recreating Test Set

from torch.utils.data import Subset

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts [0-255] RGBA to [0.0-1.0], shape [3, H, W]
    transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])  # Normalize for RGB
])

# Rebuild the original dataset
dataset = datasets.ImageFolder(root=f'sliding_spectrograms_{spectro_len}_seconds', transform=transform)

# Load indices
train_indices = torch.load(f'train_indices_v{version_number}.pt')
val_indices = torch.load(f'val_indices_v{version_number}.pt')
test_indices = torch.load(f'test_indices_v{version_number}.pt')

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

# ---------------------------------------

# --- LOADING HISTORY ---

with open(f'train_loss_history{version_number}.pkl', 'rb') as f:
    train_loss_history = pickle.load(f)
with open(f'train_acc_history{version_number}.pkl', 'rb') as f:
    train_acc_history = pickle.load(f)
with open(f'val_loss_history{version_number}.pkl', 'rb') as f:
    val_loss_history = pickle.load(f)
with open(f'val_acc_history{version_number}.pkl', 'rb') as f:
    val_acc_history = pickle.load(f)