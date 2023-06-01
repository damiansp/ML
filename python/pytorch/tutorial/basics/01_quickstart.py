import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


BATCH = 64
H = W = 28  # height and width of images
N_CATEGORIES = 10


def main():
    train, test = get_training_set()
    train_loader, test_loader = get_loaders(train, test)
    device = init_device()
    model = NeuralNetwork().to(device)
    print(model)


def get_training_set():
    return [
        datasets.FashionMNIST(
            root='data',
            train=tf,
            download=True,
            transform=ToTensor())
        for tf in [True, False]]


def get_loaders(train, test):
    loaders = [DataLoader(data, batch_size=BATCH) for data in [train, test]]
    for X, y in loaders[0]:
        print(f'Shape of X_test: [N, C, H, W]: {X.shape}')
        print(f'Shape of y_test: {y.shape} ({y.dtype})')
        break
    return loaders


def init_device():
    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu')
    print(f'Using {device} device')
    return device


class NeuralNetwork(nn.Module):
    def __init__(self):
        N_HIDDEN = 512
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(H * W, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_CATEGORIES))

    def forward(self, x):
        x = self.flatten(x)  # ? why not just call nn.Flatten() here?
        logits = self.linear_relu_stack(x)
        return logits
        


if __name__ == '__main__':
    main()
