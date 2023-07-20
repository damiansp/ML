import torch
from torchvision.models import resnet18, ResNet18_Weights

ETA = 1e-2
MOMENTUM = 0.9


def main():
    mod = resnet18(weights=ResNet18_Weights.DEFAULT)
    data = torch.rand(1, 3, 64, 64)
    labels = torch.rand(1, 1000)
    preds = mod(data)  # forward pass
    loss = (preds - labels).sum()
    loss.backward()    # backward pass
    optim = torch.optim.SGD(mod.parameters(), lr=ETA, momentum=MOMENTUM)
    optim.step()       # grad desc
    


if __name__ == '__main__':
    main()
