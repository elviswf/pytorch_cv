# -*- coding: utf-8 -*-
"""
@Time    : 2018/2/5 13:55
@Author  : Elvis
"""
"""
 basenet.py
  
"""
from torch import nn
from models.bilinearAR import BilinearAR
from torchvision.models import resnet50, resnet18
import torch.nn.functional as F


def resnetBase(num_classes=100):
    net = resnet18(pretrained=True)
    feat = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(feat, num_classes)
    )
    return net


def resnet50Base(num_classes=200):
    net = resnet50(pretrained=True)
    feat = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(feat, num_classes)
    )
    return net


class ConvBilinear(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvBilinear, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx + b
        self.bilinear1 = BilinearAR(16 * 4 * 4, 128)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        xt = x.view(x.size(0), -1)
        xt = F.relu(self.bilinear1(xt))
        out = self.fc(xt)
        return out


def bilinearNet(num_classes=10):
    return ConvBilinear(num_classes)