import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


BATCH = 4
CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
    'truck')


def main():
    train_loader, test_loader = get_data_loaders()
    show_random_images(train_loader)


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


if __name__ == '__main__':
    main()
