import numpy as np
import torch


def main():
    x_data = init_tensor()
    x_np = init_tensor_from_nparray()
    x_ones = init_ones_from_tensor(x_data)
    x_rand = init_rand_from_tensor(x_data)
    rand, ones, zeroes = init_from_shape(shape=(2,3))
    print_tensor_attributes(rand)


def init_tensor():
    data = [[1, 2], [3, 4]]
    return torch.tensor(data)


def init_tensor_from_nparray():
    np_array = np.array([[1, 2], [3, 4]])
    return torch.from_numpy(np_array)


def init_ones_from_tensor(x):
    return torch.ones_like(x)


def init_rand_from_tensor(x):
    return torch.rand_like(x, dtype=torch.float)


def init_from_shape(shape):
    rand = torch.rand(shape)
    ones = torch.ones(shape)
    zeroes = torch.zeros(shape)
    return rand, ones, zeroes


def print_tensor_attributes(x):
    print(f'Shape: {x.shape}')
    print(f'Data Type: {x.dtype}')
    print(f'Stored on: {x.device}')


if __name__ == '__main__':
    main()
          
