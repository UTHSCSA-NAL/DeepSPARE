import torch
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import nibabel as nib
import pandas as pd
from torch.utils import data
import torchvision.models as models
import math
from functools import partial


def conv1x1x1(in_channel, out_channel, stride=1):
    return nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module): 
    expansion = 4

    def __init__(self, in_channel, channel, conv_kernel, stride=1, downsample=None):
        super().__init__()
        self.conv_kernel = conv_kernel
        if self.conv_kernel == 3:
            padding = 1
        elif self.conv_kernel == 5:
            padding = 2
        elif self.conv_kernel == 7:
            padding = 3
            
        self.conv1 = conv1x1x1(in_channel, channel)
        self.bn1 = nn.BatchNorm3d(channel)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size = conv_kernel,stride = stride, padding = padding, bias=False)
        self.bn2 = nn.BatchNorm3d(channel)
        self.conv3 = conv1x1x1(channel, channel * self.expansion)
        self.bn3 = nn.BatchNorm3d(channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, channel, conv_kernel, stride=1, downsample=None):
        super().__init__()
        self.conv_kernel = conv_kernel
        if self.conv_kernel == 3:
            padding = 1
        elif self.conv_kernel == 5:
            padding = 2
        elif self.conv_kernel == 7:
            padding = 3

        self.conv1 = nn.Conv3d(in_channel, channel, kernel_size = conv_kernel,stride = stride, padding = padding, bias=False)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size = conv_kernel, stride = 1, padding = padding, bias=False)
        self.bn2 = nn.BatchNorm3d(channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MyResNet(nn.Module):
    def __init__(self, block, num_block, in_channel,conv_kernel, block_channel, num_classes, head):
        super().__init__()

        self.in_channel = in_channel
        self.num_block = num_block
        self.block_channel = block_channel
        self.conv_kernel = conv_kernel

        self.conv1 = nn.Conv3d(1, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.block1 = self.make_layer(block, block_channel[0], self.num_block[0], conv_kernel)
        if len(self.num_block) >= 2:
            self.block2 = self.make_layer(block, block_channel[1], self.num_block[1], conv_kernel,stride=2)
        if len(self.num_block) >= 3:
            self.block3 = self.make_layer(block, block_channel[2], self.num_block[2], conv_kernel,stride=2)
        if len(self.num_block) >= 4:
            self.block4 = self.make_layer(block, block_channel[3], self.num_block[3], conv_kernel,stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if head == "linear":
            self.fc = nn.Linear(self.block_channel[-1] * block.expansion, num_classes)
        if head == "mlp":
            self.fc = nn.Sequential(
                nn.Linear(self.block_channel[-1] * block.expansion, self.block_channel[-1] * block.expansion),
                nn.ReLU(inplace=True),
                nn.Linear(self.block_channel[-1] * block.expansion, num_classes))
        if head == "redp":
            self.fc = nn.Sequential(
                nn.Linear(self.block_channel[-1] * block.expansion, self.block_channel[-1] * block.expansion),
                nn.ReLU(inplace=True), nn.Dropout(),
                nn.Linear(self.block_channel[-1] * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def make_layer(self, block, channel, num_block, conv_kernel, stride=1, downsample = None):
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                    conv1x1x1(self.in_channel, channel * block.expansion, stride),
                    nn.BatchNorm3d(channel * block.expansion))

        layers = []
        layers.append(block(in_channel=self.in_channel, channel=channel, conv_kernel = conv_kernel, stride=stride, downsample=downsample))
        self.in_channel = channel * block.expansion
        for i in range(1, num_block):
            layers.append(block(self.in_channel, channel, conv_kernel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        if len(self.num_block) >= 2:
            x = self.block2(x)
        if len(self.num_block) >= 3:
            x = self.block3(x)
        if len(self.num_block) >= 4:
            x = self.block4(x)
    
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x






# def build_model(model_depth, channel, n_classes):
#     assert model_depth in [10, 18, 34, 50, 101, 152, 200]

#     if model_depth == 10:
#         model = ResNet(BasicBlock, [1, 1, 1, 1], n_classes)
#     elif model_depth == 18:
#         model = ResNet(BasicBlock, [2, 2, 2, 2], n_classes)
#     elif model_depth == 34:
#         model = ResNet(BasicBlock, [3, 4, 6, 3], n_classes)
#     elif model_depth == 50:
#         model = ResNet(Bottleneck, [3, 4, 6, 3], n_classes)
#     elif model_depth == 101:
#         model = ResNet(Bottleneck, [3, 4, 23, 3], n_classes)
#     elif model_depth == 152:
#         model = ResNet(Bottleneck, [3, 8, 36, 3], n_classes)
#     elif model_depth == 200:
#         model = ResNet(Bottleneck, [3, 24, 36, 3], n_classes)

#     return model
