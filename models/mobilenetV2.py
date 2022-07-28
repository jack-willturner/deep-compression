# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:06:35 2022

@author: hc
"""
import torch.nn as nn
from .conv_bn_relu import ConvBNReLU

class Bottleneck(nn.Module):
    """
    Bottleneck (Residual) block used mainly in MobileNet-v2.
    
    """
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super().__init__()
        self.residual_connection = (stride == 1 and in_channel == out_channel)
        # self.layers = nn.Sequential(
        #         nn.Conv2d(in_channel, in_channel * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(in_channel * expand_ratio),
        #         nn.ReLU6(inplace=True),
            
        #         nn.Conv2d(in_channel * expand_ratio, in_channel * expand_ratio, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channel * expand_ratio),
        #         nn.BatchNorm2d(in_channel * expand_ratio),
        #         nn.ReLU6(inplace=True),
            
        #         nn.Conv2d(in_channel * expand_ratio, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_channel)
        # )
        
        self.layers = nn.Sequential(
            ConvBNReLU(
            in_channel, in_channel * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False, relu=True
            ),
            ConvBNReLU(
            in_channel * expand_ratio, in_channel * expand_ratio, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channel * expand_ratio, relu=True
            ),
            ConvBNReLU(
            in_channel * expand_ratio, out_channel, kernel_size=1, stride=1, padding=0, bias=False, relu=False
            )
        )
        
        
        
        
    def forward(self, input):
        x = self.layers(input)
        if self.residual_connection:
            out = x + input
        else:
            out = x
        return out

    def get_prunable_layers(self, pruning_type="unstructured"):
        convs = []
        
        for layer in self.layers:
            convs.append(layer)

        return convs
class Conv(nn.Module):
    """
    simple convolutional layer followed by batch normalization and relu6 activation
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # self.layers = nn.Sequential(
        #         nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        #         nn.BatchNorm2d(out_channel),
        #         nn.ReLU6(inplace=True)
        #     )
        
        self.layers = ConvBNReLU(
            in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, relu=True
            )
        
    def forward(self, input):
        return self.layers(input)
    
    def get_prunable_layers(self, pruning_type="unstructured"):
        convs = []
        
        convs.append(self.layers)
        
        return convs

    
class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.num_classes = num_classes
        
        # strictly follows model architecture in mobilenet-v2 paper
        self.model = nn.Sequential(
                Conv(3, 32, stride=2),
            
                Bottleneck(32, 16, 1, 1),
                
                Bottleneck(16, 24, 2, 6),
                Bottleneck(24, 24, 1, 6),
            
                Bottleneck(24, 32, 2, 6),
                Bottleneck(32, 32, 1, 6),
                Bottleneck(32, 32, 1, 6),
            
                Bottleneck(32, 64, 2, 6),
                Bottleneck(64, 64, 1, 6),
                Bottleneck(64, 64, 1, 6),
                Bottleneck(64, 64, 1, 6),
            
                Bottleneck(64, 96, 1, 6),
                Bottleneck(96, 96, 1, 6),
                Bottleneck(96, 96, 1, 6),
            
                Bottleneck(96, 160, 2, 6),
                Bottleneck(160, 160, 1, 6),
                Bottleneck(160, 160, 1, 6),
            
                Bottleneck(160, 320, 1, 6),
            
                Conv(320, 1280, kernel_size=1, stride=1, padding=0)
        )
        
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # instead of flattening and then using linear layer, used Conv2d then flattened
        #  
        self.fc = nn.Conv2d(1280, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, input):
        x = self.model(input)
        x = self.avg_pool(x)
        x = self.fc(x)
        out = x.view(-1, self.num_classes)
        return out
    
    def get_prunable_layers(self, pruning_type="unstructured"):
        convs = []

        for layer in self.model:
            if isinstance(layer,Conv):
                for conv in layer.get_prunable_layers(pruning_type):
                    convs.append(conv)
            elif isinstance(layer,Bottleneck):
                for conv in layer.get_prunable_layers(pruning_type):
                    convs.append(conv)

        return convs
    
def mobilenet2():
    return MobileNetV2(10)