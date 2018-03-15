# -*- coding: utf-8 -*-
"""
@Time    : 2018/2/5 13:55
@Author  : Elvis
"""
"""
 basenet.py
  
"""
from torch import nn
from torchvision.models import resnet50, resnet18
import torch.nn.functional as F


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                m.bias.data.zero_()


def resnetBase(num_classes=100):
    net = resnet18(pretrained=True)
    feat = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(feat, num_classes)
    )
    return net


def resnet18w(num_classes=100):
    net = resnet18(pretrained=False)
    feat = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(feat, 512),
        nn.ReLU(inplace=True),
        nn.Linear(feat, num_classes, bias=False)
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


class ConvBase(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvBase, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def conv2w(num_classes=10):
    net = ConvBase(num_classes)
    init_weights(net)
    return net
