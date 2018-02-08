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
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torchvision.models import resnet18, resnet50


class BilinearAR(nn.Module):
    def __init__(self, in_features, out_features):
        super(BilinearAR, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)

        self.fc2 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        attr = self.fc1(x)
        wt = self.fc2(x)
        xt = wt.mul(attr)
        return xt


class NeWNet(nn.Module):
    def __init__(self, cnn, num_feat=1024, num_classes=200):
        super(NeWNet, self).__init__()
        self.feat_size = cnn.fc.in_features
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])

        self.bilinear1 = BilinearAR(self.feat_size, num_classes)

    def forward(self, x):
        x1 = self.cnn(x)
        x1 = x1.view(x1.size(0), -1)
        xt = self.bilinear1(x1)
        return xt


class BilinearNet(nn.Module):
    def __init__(self, cnn, num_feat=1024, num_classes=200):
        super(BilinearNet, self).__init__()
        self.feat_size = cnn.fc.in_features
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])

        self.bilinear1 = BilinearAR(self.feat_size, num_feat)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_feat, num_classes)
        )

    def forward(self, x):
        x1 = self.cnn(x)
        x1 = x1.view(x1.size(0), -1)
        xt = self.bilinear1(x1)
        out = self.fc(xt)
        return out


class Bilinear2Net(nn.Module):
    def __init__(self, cnn, num_feat=1024, num_classes=200):
        super(Bilinear2Net, self).__init__()
        self.feat_size = cnn.fc.in_features
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        self.bilinear1 = BilinearAR(self.feat_size, num_feat)
        self.bilinear2 = BilinearAR(num_feat, 512)
        self.fc = nn.Linear(512, num_classes)
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes)
        # )

    def forward(self, x):
        x1 = self.cnn(x)
        x1 = x1.view(x1.size(0), -1)
        xt = self.bilinear1(x1)
        xt = self.bilinear2(xt)
        out = self.fc(xt)
        return out


def resnetAR(num_classes=200):
    cnn = resnet50(pretrained=True)
    return BilinearNet(cnn, num_feat=1024, num_classes=num_classes)


def resnet2AR(num_classes=200):
    cnn = resnet50(pretrained=True)
    return Bilinear2Net(cnn, num_feat=1024, num_classes=num_classes)


def resnet18AR(num_classes=200):
    cnn = resnet18(pretrained=True)
    return NeWNet(cnn, num_feat=512, num_classes=num_classes)


def resnet18ARfc(num_classes=200):
    cnn = resnet18(pretrained=True)
    return BilinearNet(cnn, num_feat=512, num_classes=num_classes)


class NeuralWeightedUnit(nn.Module):
    def __init__(self, layer, in_features, out_features):
        super(NeuralWeightedUnit, self).__init__()
        self.layer = layer

        self.fc2 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
            nn.Dropout(0.5)
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        attr = self.layer(x)
        wt = self.fc2(x)
        xt = wt.mul(attr)
        out = self.activation(xt)
        return out