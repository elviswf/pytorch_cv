# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/3 19:50
@Author  : Elvis
"""
"""
 fusion.py
  
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, alexnet


class FusionNet(nn.Module):
    def __init__(self, cnn0, cnn1, num_classes=200):
        super(FusionNet, self).__init__()
        self.cnn0 = cnn0
        self.cnn1 = cnn1
        # self.fc0 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes)
        # )
        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x0 = self.cnn0(x)
        x1 = self.cnn1(x)
        xn = x0.squeeze() + x1.squeeze()
        out = self.fc1(xn)
        return out, x1


def fusionResnet(num_classes=200):
    cnn0 = resnet18(pretrained=True)
    cnn0 = nn.Sequential(*list(cnn0.children())[:-1])
    cnn1 = resnet18(pretrained=False)
    cnn1 = nn.Sequential(*list(cnn1.children())[:-1])
    return FusionNet(cnn0, cnn1, num_classes)


class FusionLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(FusionLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_true, y_pred, x1):
        # convert logits to soft targets
        return F.cross_entropy(y_true, y_pred) + self.alpha * x1.norm(1)
