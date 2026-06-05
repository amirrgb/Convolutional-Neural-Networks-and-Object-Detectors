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


def get_mnist_data_with_noise(batch_size=64, noise_std=0.5):
    current_dir = Path(__file__).parent  # .../src
    data_root = current_dir.parent / 'data'  # .../data
    
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


def generate_sine_cosine_data(num_samples=5000, signal_length=100, noise_std=0.1):
    """
    Generate binary classification dataset.
    Class 0: sine wave, Class 1: cosine wave.
    Each sample is a 1D signal of length `signal_length`.
    """
    X = []
    y = []
    t = np.linspace(0, 4 * np.pi, signal_length)  # frequency: 2 cycles over length 100
    for i in range(num_samples):
        if i < num_samples // 2:
            # Sine wave (class 0)
            signal = np.sin(t)
            label = 0
        else:
            # Cosine wave (class 1)
            signal = np.cos(t)
            label = 1
        # Add Gaussian noise
        signal += np.random.normal(0, noise_std, signal_length)
        X.append(signal)
        y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    return X, y

def generate_hard_sine_cosine_data(num_samples=5000, signal_length=100, 
                                   freq_range=(1, 3), phase_range=(0, 2*np.pi),
                                   noise_std=0.5):
    """
    Generate sine vs cosine classification with random frequency and phase per sample.
    Class 0: sine wave, Class 1: cosine wave.
    """
    X = []
    y = []
    for i in range(num_samples):
        # Random frequency (cycles per signal_length)
        freq = np.random.uniform(*freq_range)
        t = np.linspace(0, 2 * np.pi * freq, signal_length)
        # Random phase shift (same for both sine and cosine to keep relation)
        phase = np.random.uniform(*phase_range)
        if i < num_samples // 2:
            # Sine wave (class 0)
            signal = np.sin(t + phase)
            label = 0
        else:
            # Cosine wave (class 1)
            signal = np.cos(t + phase)
            label = 1
        # Add Gaussian noise
        signal += np.random.normal(0, noise_std, signal_length)
        X.append(signal)
        y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    return X, y