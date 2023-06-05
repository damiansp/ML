import numpy as np
import torch


def main():
    x_data = init_tensor()
    x_np = init_tensor_from_nparray()
    x_ones = init_ones_from_tensor(x_data)
    x_rand = init_rand_from_tensor(x_data)
    rand, ones, zeroes = init_from_shape(shape=(2,3))
    print_tensor_attributes(rand)
    if torch.cuda.is_available():
        rand = rand.to('cuda')
    rand = torch.rand(4, 4)
    slice_ops(rand)
    join_tensors(rand, rand)
    arithmetic(rand)


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


def slice_ops(x):
    print(f'First row: {x[0]}')
    print(f'First col: {x[:, 0]}')
    print(f'Last col: {x[:, -1]}')
    x[:, 1] = 0
    print(x)

    
def join_tensors(x, y):
    joined = torch.cat([x, y], dim=1)
    print(joined)


def arithmentic(x):
    xxT = x @ x.T  # same as
    xxT = tensor.matmul(x.T)
    z = x * x  # same as
    w = x.mul(x)
    agg = tensor.sum()
    print(agg.item())
    # inplace ops
    x.add_(5)
    
if __name__ == '__main__':
    main()
          
