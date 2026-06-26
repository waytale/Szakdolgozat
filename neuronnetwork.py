
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#requires_grad = True 





class SimpleMLP(nn.Module):
    def __init__(self, in_dim=784, hidden=64, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def dataset():
    """Loads the MNIST dataset and returns data loaders for training and testing."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)
    return train_ds, test_ds, train_loader, test_loader


def print_dataset_info():
    """Prints information about the MNIST dataset and data loaders."""    
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Train loader size: {len(train_loader)}")
    print(f"Test loader size: {len(test_loader)}")
    print(f"Input shape: {train_ds[0][0].shape}")
    print(f"Label shape: {train_ds[0][1]}")
    print(f"Input shape (flattened): {train_ds[0][0].view(-1).shape}")



train_ds, test_ds, train_loader, test_loader = dataset()
print_dataset_info()

