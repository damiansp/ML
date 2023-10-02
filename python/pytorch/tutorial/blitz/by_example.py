from math import pi

import matplotlib.pyplot as plt
import numpy as np
import torch


ETA = 1e-6
EPOCHS = 2_000
N = 2_000
DTYPE = torch.float
DEVICE = torch.device('cpu')


def main():
    warm_up_with_numpy()
    use_tensors()


def warm_up_with_numpy():
    a, b, c, d = train_np()
    print(f'model: {a:.4f} + {b:.4f}x + {c:.4f}x^2 + {d:.4f}x^3')


def train_np():
    x, y = make_inputs()
    a, b, c, d = init_weights(4)
    for epoch in range(EPOCHS):
        # model to predict: y = a + bx + cx^2 + dx^3
        preds = a + b*x + c*x**2 + d*x**3
        loss = get_sse(y, preds)
        if epoch % 100 == 99:
            print(f'Epoch {epoch}: loss {loss:.5f}')
        # backprop
        grad = get_partial_derivatives(x, y, preds)
        a, b, c, d = [w - ETA*d for w, d in zip([a, b, c, d], grad)]
    plt.scatter(x, y, alpha=0.05, label='data')
    fitted = a + b*x + c*x**2 + d*x**3
    plt.plot(x, fitted, color='r', label='fitted')
    plt.legend()
    plt.show()
    return a, b, c, d
                

def make_inputs():
    x = np.linspace(-pi, pi, N)
    y = np.sin(x)
    return x, y


def init_weights(n):
    return [np.random.randn() for _ in range(n)]


def get_sse(y, preds):
    return np.square(preds - y).sum()


def get_partial_derivatives(x, y, preds):
    dpred = 2. * (preds - y)
    da = dpred.sum()
    db = (dpred * x).sum()
    dc = (dpred * x**2).sum()
    dd = (dpred * x**3).sum()
    return [da, db, dc, dd]


def use_tensors():
    a, b, c, d = train_torch()
    print(f'model: {a:.4f} + {b:.4f}x + {c:.4f}x^2 + {d:.4f}x^3')
    

def train_torch():
    x, y = make_torch_inputs()
    a, b, c, d = init_torch_weights(4)
    for epoch in range(EPOCHS):
        preds = a + b*x + c*x**2 + d*x**3
        loss = get_sse_torch(y, preds)
        if epoch % 100 == 99:
            print(f'Epoch {epoch}: loss {loss:.5f}')
        # backprop
        grad = get_partial_derivatives(x, y, preds)  # same
        a, b, c, d = [w - ETA*d for w, d in zip([a, b, c, d], grad)]
    plt.scatter(x, y, alpha=0.05, label='data')
    fitted = a + b*x + c*x**2 + d*x**3
    plt.plot(x, fitted, color='r', label='fitted')
    plt.legend()
    plt.show()
    return a, b, c, d


def make_torch_inputs():
    x = torch.linspace(-pi, pi, N)
    y = torch.sin(x)
    return x, y


def init_torch_weights(n):
    return [torch.randn((), device=DEVICE, dtype=DTYPE) for _ in range(n)]


def get_sse_torch(y, preds):
    return (preds - y).pow(2).sum().item()


    

if __name__ == '__main__':
    main()
