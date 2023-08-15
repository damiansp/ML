import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_IMG_CHANNELS = 1
OUTPUT_CHANNELS1 = 6
OUTPUT_CHANNELS2 = 16
CONV_DIM = 5
HIDDEN1 = 120
HIDDEN2 = 84
N_LABELS = 10
MAX_POOL_DIM = (2, 2)  # just 2 ok if square

        
def main():
    net = Net()
    print(net)
    params = list(net.parameters())
    print('n params:', len(params))
    print('conv1 .weight:', params[0].size())  # 6, 1, 5, 5
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)
    net.zero_grad()
    out.backward(torch, randn(1, 10))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_IMG_CHANNELS, OUTPUT_CHANNELS1, CONV_DIM)
        self.conv2 = nn.Conv2d(OUTPUT_CHANNELS1, OUTPUT_CHANNELS2, CONV_DIM)
        # y = Wx + b
        self.fc1 = nn.Linear(OUTPUT_CHANNELS2 * CONV_DIM * CONV_DIM, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, N_LABELS)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), MAX_POOL_DIM)
        x = F.max_pool2d(F.relu(self.conv2(x)), MAX_POOL_DIM)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    
