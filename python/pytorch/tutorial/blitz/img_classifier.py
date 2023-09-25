import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


BATCH = 4
CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
    'truck')
ETA = 0.001
MOMENTUM = 0.9
EPOCHS = 2
MOD_PATH = './cifar_net.pth'


def main():
    train_loader, test_loader = get_data_loaders()
    show_random_images(train_loader)
    net = Net()
    net = train(train_loader, net)
    test(test_loader, net)


def get_data_loaders():
    print('Loading data...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set, test_set = [
        torchvision.datasets.CIFAR10(
            root='./data', train=TF, download=True, transform=transform)
        for TF in [True, False]]
    train_loader, test_loader = [
        torch.utils.data.DataLoader(
            data_set, batch_size=BATCH, shuffle=TF, num_workers=2)
        for data_set, TF in zip([train_set, test_set], [True, False])]
    return train_loader, test_loader


def show_random_images(train_loader):
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    show_img(torchvision.utils.make_grid(images))
    print(' '.join(f'{CLASSES[labels[i]]:5s}' for i in range(BATCH)))


def show_img(img):
    img = img/2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(CLASSES))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(train_loader, net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=ETA, momentum=MOMENTUM)
    for epoch in range(EPOCHS):
        running_loss = 0.
        for i, data in enumerate(train_loader, 0): # 0 is redundant/default
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2_000 = 1999:
                print(
                    f'[{epoch + 1}, {i + 1:5d}] '
                    f'loss: {running_loss / 2000:.3f}')
                running_loss = 0.
    print('Training complete')
    torch.save(net.state_dict(), MOD_PATH)
    print('Model saved to', MOD_PATH)
    return net


def test(test_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(MOD_PATH))
    test_sample(test_loader, net, device)
    test_all(test_loader, net, device)
    assess_performance(test_loader)
    
    
def test_sample(test_loader, net, device)
    data_iter = iter(test_loader)
    imgs, labels = next(data_iter)
    imgs = imgs.to(device)
    labels = labels.to(device)
    show_img(torchvision.utils.make_grid(imgs))
    print(
        'Ground Truth:',
        ' '.join(f'{CLASSES[labels[i]]:5s}' for i in range(4)))
    outputs = net(imgs)
    _, preds = torch.max(outputs, 1)
    print('Predicted:', ' '.join(f'{CLASSES[preds[i]]:5s}' for i in range(4)))


def test_all(test_loader, net, device):
    n_correct = 0
    n = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = net(imgs)
            _, preds = torch.max(output.data, 1)
            n += labels.size(0)
            n_correct += (preds == labels).sum().item()
    print(f'Accuracy: {n_correct / n}')


def assess_performance(test_loader):
    correct_pred = {c: 0 for c in CLASSES}
    total_pred = {c: 0 for c in CLASSES}
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            outputs = net(imgs)
            _, preds = torch.max(output.data, 1)
            for label, pred in in zip(labels, preds):
                if label == pred:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1
    print('Accuracy by class:')
    for classname, n_correct in correct_pred.items():
        acc = n_correct / total_pred[classname]
        print(f'{classname:5s}: {acc:.2f}')
    
if __name__ == '__main__':
    main()
