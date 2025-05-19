import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.transforms import v2
import sys
import time 
#Output Height/Width = [(Input Size - Kernel Size + 2 * Padding) / Stride] + 1

image_shape = (3, 256,256)
input = torch.randn(1,*image_shape)
"""
resnet_transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.CenterCrop(224),
   # v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
"""
class ViT(nn.Module):
    def __init__(self, pretrained=True, freeze_layers = True):
        super().__init__()
        self.freeze = freeze_layers
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.body = models.vit_b_16(weights=weights)
        if freeze_layers:
            for param in self.body.parameters():
                param.requires_grad = False
        self.body.heads.head = nn.Identity()
        self.fc = nn.Sequential(nn.Linear(768, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 5))
        print([name for name, param in self.body.named_parameters() if param.requires_grad])
        print([name for name, param in self.fc.named_parameters() if param.requires_grad])

    def forward(self, x):
        x = self.body(x)
        x = self.fc(x)
        return x
    
class Inceptionv3(nn.Module):
    def __init__(self, pretrained=True, freeze_layers = True, top_layer=True, aux_logits=True):
        super().__init__()
        self.freeze = freeze_layers
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        self.body = models.inception_v3(weights=weights)
        self.aux_logits = aux_logits
        self.top_layer = top_layer
        
        if freeze_layers:
            for param in self.body.parameters():
                param.requires_grad = False
        if not aux_logits:
            self.body.aux_logits = False
        if top_layer:
            self.fc = nn.ModuleList([nn.Linear(1000,5), nn.Linear(1000,5)])
        else:
            self.body.fc = nn.Linaer(2048,5)
            self.body.AuxLogits.fc = nn.Linear(768,5)
       
    def forward(self, x):
        x = self.body(x) #named tuple 2x[batch_size, 1000]
        if self.aux_logits and self.training:
            if self.top_layer:
                return self.fc[0](x[0]), self.fc[1](x[1])
            return x[0], x[1]
        else:
            if self.top_layer:
                return self.fc[0](x)
            return x
        


class ResNet(nn.Module):
    def __init__(self,type='101', pretrained=True, freeze_layers = True):
        super().__init__()
        self.freeze = freeze_layers
        weights = 'DEFAULT' if pretrained else None
        self.body = getattr(models, f"resnet{type}")(weights=weights)
        if freeze_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False
        in_features = self.body.fc.in_features
        out_features = self.body.fc.out_features
        self.body.fc = nn.Identity()
        self.fc = nn.Sequential(nn.Linear(in_features, 1024),
                                nn.GELU(),
                                nn.Linear(1024, 5))
        #self.fc = nn.Linear(out_features,5)
        #print([name for name, param in self.resnet.named_parameters() if param.requires_grad])
    def forward(self, x):
        x = self.body(x)
        x = self.fc(x)
        return x
    #
class SimpleBlock(nn.Module):
    conv_params = {'kernel_size': 3, 'stride': 1, 'padding': 1}
    def __init__(self, in_fan, out_fan, conv_params=None, dropout=0, pool=True, batchnorm=True):
        super().__init__()
        if conv_params is None:
            conv_params = self.conv_params
        self.block = nn.ModuleList()
        self.conv = nn.Conv2d(in_fan, out_fan, **conv_params)
        self.block.extend([self.conv, nn.ReLU()])
        if batchnorm:
            self.block.append(nn.BatchNorm2d(out_fan))
        if dropout:
            self.block.append(nn.Dropout(self.dropout))
        if pool:
            self.block.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        

    def forward(self,x):
        for module in self.block:
            x = module(x)
        return x

#Simple baseline
class FlowerModel(nn.Module):
    def __init__(self, n_blocks=4, start_channels=32, adaptive_pool_size=2):
        super().__init__()
        channels = [3] + [start_channels * (2**i) for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            SimpleBlock(channels[i], channels[i+1]) 
            for i in range(n_blocks)
        ])
        self.classifier = nn.Linear(channels[-1]*adaptive_pool_size**2, 5)
        self.conv = nn.Conv2d(channels[-1], 256, kernel_size=3,stride=2,padding=1)
        self.global_pool = nn.AdaptiveMaxPool2d(adaptive_pool_size)

        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.conv(x)
        x = self.global_pool(x)
        #print(x.shape)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    #resnet = ResNet(pretrained=True, freeze_layers=True)
    model = FlowerModel(n_blocks=4, start_channels=32)
    with torch.no_grad():
        t = time.time()
        print(model(input).shape)
        print('Elapsed time:', time.time()-t)
    
