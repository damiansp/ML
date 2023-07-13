import numpy as np
import torch


def main():
    rand_t = init_tensors()
    show_tensor_attributes(rand_t)
    tensor_ops(rand_t)


def init_tensors():
    data = [[1, 2], [3, 4]]
    x_data = init_tensor_from_data(data)
    x_np = init_tensor_from_numpy(data)
    x_ones, x_rand = init_tensor_from_tensor(x_data)
    rand_t, ones_t, zeros_t = init_tensor_with_rand_or_const_vals()
    return rand_t


def init_tensor_from_data(data):
    x_data = torch.tensor(data)
    return x_data


def init_tensor_from_numpy(data):
    np_data = np.array(data)
    x_np = torch.from_numpy(np_data)
    return x_np


def init_tensor_from_tensor(tensor):
    x_ones = torch.ones_like(tensor)
    x_rand = torch.rand_like(tensor, dtype=torch.float)
    return x_ones, x_rand


def init_tensor_with_rand_or_const_vals():
    shape = (2, 3)
    rand_t = torch.rand(shape)
    ones_t = torch.ones(shape)
    zeros_t = torch.zeros(shape)
    return rand_t, ones_t, zeros_t


def show_tensor_attributes(tensor):
    print(f'Shape: {tensor.shape}')
    print(f'DType: {tensor.dtype}')
    print(f'Stored on: {tensor.device}')


def tensor_ops(tensor):
    tensor = move_to_gpu(tensor)
    index_tensor(tensor)
    big_tensor = join_tensors(tensor, tensor)


def move_to_gpu(tensor):
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')
        print('Tensor moved to GPU')
    else:
        print('Tensor on CPU')
    return tensor


def index_tensor(tensor):
    tensor = torch.ones(4, 4)
    tensor[:, 1] = 0
    print('tensor:', tensor)


def join_tensors(*args):
    t = torch.cat([*args], dim=1)
    print('Concat:', t)
    return t

if __name__ == '__main__':
    main()
