import torch
from torch import nn, optim
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
    differentiate()
    exclude_from_dag()
    freeze_for_fine_tuning()


def differentiate():
    a = torch.tensor([2., 3.], requires_grad=True)
    b = torch.tensor([6., 4.], requires_grad=True)
    Q = 3 * a**3 - b**2
    external_grad = torch.tensor(1., 1.])
    Q.backward(gradient=external_grad)
    print(a.grad == 9 * a**2)  # True
    print(b.grad == -2 * b)    # True
    

def exclude_from_dag():
    x = torch.rand(5, 5)
    y = torch.rand(5, 5)
    z = torch.rand((5, 5), requires_grad=True)
    a = x + y
    b = x + z
    print('a requires grad:', a.requires_grad)  # F
    print('b requires grad:', b.requires_grad)  # T


def freeze_for_fine_tuning():
    mod = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Freeze all params
    for param in mod.parameters():
        param.requires_grad = False
    mod.fc = nn.Linear(512, 10)   # refit top layer to new labels
    optimizer = optim.SGD(mod.parameters(), lr=ETA, momentum=MOMENTUM)


if __name__ == '__main__':
    main()
