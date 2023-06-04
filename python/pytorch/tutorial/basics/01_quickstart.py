import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


CLASSES = [
    'T-shirt/top', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
    'Sneaker', 'Bag', 'Ankle boot']
H = W = 28  # height and width of images
N_CLASSES = len(CLASSES)
BATCH = 64
ETA = 1e-3
EPOCHS = 5
PRINT_EVERY_N = 100


def init_device():
    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu')
    print(f'Using {device} device')
    return device


device = init_device()

    
def main():
    train, test = get_training_set()
    train_loader, test_loader = get_loaders(train, test)
    model = NeuralNetwork().to(device)
    print(model)
    loss_func = nn.CrossEntropyLoss()
    for e in range(EPOCHS):
        print(f'Epoch {e + 1}---------------------------------------')
        train_model(train_loader, model, loss_func)
        test_model(test_loader, model, loss_func)
    print('Complete')
    model_path = 'model.pth'
    save_model(model, model_path)
    model = load_model(model_path)
    model.eval()
    x, y = test[0]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = CLASSES[pred[0].argmax(0)], CLASSES[y]
        print(f'Predicted: "{predicted}"; Acutal: "{actual}"')
    

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
            nn.Linear(N_HIDDEN, N_CLASSES))

    def forward(self, x):
        x = self.flatten(x)  # ? why not just call nn.Flatten() here?
        logits = self.linear_relu_stack(x)
        return logits


def train_model(loader, model, loss_func):
    size = len(loader.dataset)
    optimizer = torch.optim.SGD(model.parameters(), lr=ETA)
    model.train()
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_func(pred, y)
        loss.backward()  # backprop
        optimizer.step()
        optimizer.zero_grad()
        if batch % PRINT_EVERY_N == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'Loss: {loss:>7f} [{current:>5d} / {size:>5d}]')


def test_model(loader, model, loss_func):
    size = len(loader.dataset)
    n_batches = len(loader)
    model.eval()
    test_loss, n_correct = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            n_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= n_batches
    n_correct /= size
    print(
        f'Test Err:\n'
        f'  Acc: {(100 * n_correct):>0.1}%\n'
        f'  Avg loss: {test_loss:>8f}\n')


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Saved PyTorch model to "{path}"')


def load_model(path):
    print(f'Loading model from "{path}"')
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(path))
    return model
    

if __name__ == '__main__':
    main()
