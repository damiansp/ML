import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


ETA = 1e-3
BATCH = 64
EPOCHS = 10


def main():
    training_data, test_data = load_data()
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)
    mod = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mod.parameters(), lr=ETA)
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}\n------------------------------------------')
        train(train_dataloader, mod, loss_fn, optimizer)
        test(test_dataloader, mod, loss_fn)
    print('Done!')

    
def load_data():
    return [
        datasets.FashionMNIST(
            root='data',
            train=tf,
            download=True,
            transform=ToTensor())
        for tf in [True, False]]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits




def train(dataloader, mod, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set mod to training - imoprtant for batch norm and dropout;
    # unnecessary here
    mod.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = mod(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'Loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def test(dataloader, mod, loss_fn):
    # Also unnecessary here:
    mod.eval()
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    test_loss, correct = 0, 0
    # Eval mod with no_grad prevents gradients from being computed
    with torch.no_grad():
        for X, y in dataloader:
            pred = mod(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= n_batches
        correct /= size
        print(
            f'Test err:\n Acc: {100 * correct:>0.1f}%; '
            f'mean loss: {test_loss:>8f}\n')
        
        
if __name__ == '__main__':
    main()
