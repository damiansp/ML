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
    print(model)
    for name, param, in mod.named_parameters():
        print(f'Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n')


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
    layer1 = nn.Linear(in_features=28 * 28, out_features=20)
    hidden1 = layer1(flat_img)
    print(hidden1.size())    # 3, 20
    h1 = nn.ReLU()(hidden1)  # 0s out neg values
    
    seq = nn.Sequential(flatten, layer1, nn.ReLU(), nn.Linear(20, 10))
    logits = seq(input_img)
    softmax = nn.Softmax(dim=2)
    pred_probs = softmax(logits)
                         
                        
