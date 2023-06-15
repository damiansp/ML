import torch
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor


def main():
    ds = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),  # PIL image or np.ndarray to FloatTensor on [0, 1]
        target_transform = Lambda(
            lambda y: torch.zeros(
                10, dtype=torch.float).scatter(0, torch.tensor(y), value=1)))
    


if __name__ == '__main__':
    main()
