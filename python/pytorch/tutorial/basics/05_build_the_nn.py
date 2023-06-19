import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    device = (
        'cuda' if torch.cuda.is_available
        else 'mps' if torch.backend.mps.is_available else 'cpu')
    mod = NeuralNetwork().to(device)
    print(mod)
    x = torch.rand(1, 28, 28, device=device)
    logits = mod(X)
    pred_probs = nn.Softmax(dim=1)(logits)
    preds = pred_probs.argmax(1)
    print(f'Predicted class:', preds)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # why?
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def spell_it_out():
    input_img = torch.rand(3, 28, 28)
    print(input_img.size())  # 3, 28, 28
    flatten = nn.Flatten()
    flat_img = flatten(input_img)
    print(flat_img.size())   # 3, 784
