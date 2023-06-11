import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.io import read_image
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
BATCH = 64


def main():
    train, test = load_train_test()
    plot_preview(train)
    train_loader, test_loader = [
        DataLoader(dset, batch_size=BATCH, shuffle=True) for dset in [train, test]]
    # iterate through loader
    train_X, train_y = next(iter(train_loader))
    print(f'X batch shape: {train_X.size()}')
    print(f'y batch shape: {train_y.size()}')
    img = train_X[0].squeeze()
    label = train_y[0]
    plt.imshow(img, cmap='gray')
    plt.show()
    print(f'Label: {LABELS[label.item()]}')


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


class CustomImageDataset(Dataset):
    def __init__(
            self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            img = self.transform(img)
        if self.target_tranform:
            label = self.target_transform(label)
        return img, label


if __name__ == '__main__':
    main()
