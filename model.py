import torch
from torch import nn
from torch.nn import functional as F
#Output Height/Width = [(Input Size - Kernel Size + 2 * Padding) / Stride] + 1

image_shape = (3,256,256)
input = torch.randn(1,*image_shape)
print(input.shape)

class SimpleBlock(nn.Module):
    conv_params = {'kernel_size': 3, 'stride': 1, 'padding': 1}
    def __init__(self, in_fan, out_fan, conv_params=None, dropout=0, pool=True, batchnorm=True):
        super().__init__()
        if conv_params is None:
            conv_params = self.conv_params
        
        self.conv = nn.Conv2d(in_fan, out_fan, **conv_params)
        self.dropout = dropout
        self.pool = pool
        self.batchnorm = batchnorm
        if self.dropout:
            self.dropout = nn.Dropout(self.dropout)
        if self.pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        if self.batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_fan)

    def forward(self,x):
        x = F.relu(self.conv(x))
        if self.batchnorm:
            x = self.batchnorm(x)
        if self.pool:
            x = self.pool(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class FlowerModel(nn.Module):
    def __init__(self, n_blocks=4, start_channels=32):
        super().__init__()
        channels = [3] + [start_channels * (2**i) for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            SimpleBlock(channels[i], channels[i+1]) 
            for i in range(n_blocks)
        ])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        return x
    

if __name__ == "__main__":
    # Test the FlowerModel
    input = torch.randn(1, 3, 256, 256)
    model = FlowerModel()
    print(model(input).shape)
    block = SimpleBlock(3, 32)
    print(block(input).shape)
    
