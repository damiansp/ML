import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


DATA = './data'
LABELS = {
    0: 'T-shirt',
    1: 'Pants',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'}
N_CLASSES = len(LABELS)


def main():
    train, test = load_train_test()
    plot_preview(train)


def load_train_test():
    return [
        datasets.FashionMNIST(
            root=DATA, train=tf, download=True, transform=ToTensor())
        for tf in (True, False)]


def plot_preview(data):
    fig = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    n = len(data)
    for i in range(1, cols*rows + 1):
        idx = torch.randint(n, size=(1,)).item()
        img, label = data[idx]
        fig.add_subplot(rows, cols, i)
        plt.title(LABELS[label])
        plt.axis('off')
        plt.imshow(img.squeeze(), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
