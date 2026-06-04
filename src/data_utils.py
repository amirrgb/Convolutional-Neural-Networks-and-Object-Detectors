import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


def get_regression_data(test_size=0.25, noise_std=0.2, batch_size=64, random_state=42):
    np.random.seed(random_state)
    X = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, noise_std, X.shape[0])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # تبدیل به تنسور
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    print("Train shape (regression):", X_train.shape)
    return X_train, X_test, y_train, y_test, train_loader, test_loader



def get_moons_data(n_samples=400, noise=0.1, test_size=0.2, batch_size=64, random_state=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    y = y.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    print("Train shape (moons):", X_train.shape)
    return X_train, X_test, y_train, y_test, train_loader, test_loader



def get_mnist_data(batch_size=64, random_state=42):
    current_dir = Path(__file__).parent  # .../src
    data_root = current_dir.parent / 'data'  # .../data
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = torchvision.datasets.MNIST(
        root=str(data_root), train=True,
        download=False, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=str(data_root), train=False,
        download=False, transform=transform
    )
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    X_train = trainset.data.numpy()      # (60000, 28, 28)
    y_train = trainset.targets.numpy()   # (60000,)
    X_test = testset.data.numpy()
    y_test = testset.targets.numpy()
    
    return X_train, X_test, y_train, y_test, trainloader, testloader



class NoisyMNIST(Dataset):
    def __init__(self, dataset, noise_std=0.5):
        self.dataset = dataset
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        noise = torch.randn_like(img) * self.noise_std
        img_noisy = img + noise
        return img_noisy, label


def get_mnist_data_with_noise(batch_size=64, noise_std=0.5, data_root=None):
    if data_root is None:
        try:
            current_dir = Path(__file__).parent
            data_root = current_dir.parent / 'data'
        except NameError:
            data_root = Path.cwd() / 'data'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset_clean = torchvision.datasets.MNIST(root=str(data_root), train=True,
                                                download=False, transform=transform)
    testset_clean = torchvision.datasets.MNIST(root=str(data_root), train=False,
                                               download=False, transform=transform)
    trainset_noisy = NoisyMNIST(trainset_clean, noise_std=noise_std)
    testloader = DataLoader(testset_clean, batch_size=batch_size, shuffle=False)
    trainloader_noisy = DataLoader(trainset_noisy, batch_size=batch_size, shuffle=True)
    return trainloader_noisy, testloader
