# -*- coding: utf-8 -*-
"""
@Time    : 2018/2/4 20:35
@Author  : Elvis
"""
"""
 bilinearAR.py
  
"""
import torch
from torch import nn
from torch.autograd import Variable, Function
from torchvision.models import resnet18, resnet50


class BilinearAR(nn.Module):
    def __init__(self, in_features, out_features):
        super(BilinearAR, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        # orthogonal(self.fc0.weight)

        self.fc2 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        wt = self.fc1(x)
        attr = self.fc2(x)
        xt = attr.mul(wt)
        return xt


class BilinearNet(nn.Module):
    def __init__(self, cnn, num_classes=200):
        super(BilinearNet, self).__init__()
        self.feat_size = cnn.fc.in_features
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        self.bilinear1 = BilinearAR(self.feat_size, 1024)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x1 = self.cnn(x)
        x1 = x1.view(x1.size(0), -1)
        xt = self.bilinear1(x1)
        out = self.fc(xt)
        return out


def resnetAR(num_classes=200):
    cnn = resnet50(pretrained=True)
    return BilinearNet(cnn, num_classes)
