import torch


def main():
    x, y = init_vars()
    w, b = init_weights()
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    print(f'Grad func for z: {z.grad_fn}')
    print(f'Grad func for loss: {loss.grad_fn}')
    loss.backward()
    print(w.grad)
    print(b.grad)
    print(z.requires_grad)          # True
    with torch.no_grad():
        z = torch.matmul(x, w) + b  # e.g., just to print some intermed preds
        print(z.requires_grad)      # False
    print(z.requires_grad)          # False
    z_det = z.detach()
    print(z_det.requires_grad)      # False
    tensor_gradient()


def init_vars():
    x = torch.ones(5)
    y = torch.zeros(3)
    return x, y


def init_weights():
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    return w, b


def tensor_gradient():
    inp = torch.eye(4, 5, requires_grad=True)
    out = (inp + 1).pow(2).t()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f'First call:\n{inp.grad}')
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f'Second call:\n{inp.grad}')
    inp.grad.zero_()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f'After zeroing gradient:\n{inp.grad}')
    

if __name__ == '__main__':
    main()
