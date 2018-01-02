# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/3 19:50
@Author  : Elvis
"""
"""
 emb_resnet.py
  
"""
import torch
from torch import nn
from torchvision.models import resnet18, alexnet


class Bilinear(nn.Module):
    def __init__(self, cnn1, cnn2, num_classes=200):
        super(Bilinear, self).__init__()
        self.cnn1 = cnn1
        self.cnn2 = cnn2
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*512, num_classes)
        )

    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x1 = x1.view(x1.size(0), 512, -1)
        x2 = x2.view(x2.size(0), 512, -1)
        x2t = x2.transpose(1, 2)
        feat = x1.matmul(x2t) / x1.size(-1)
        feat = feat.view(feat.size(0), -1)
        fsqrt = torch.sign(feat) * torch.sqrt(feat.abs()+1e-12)
        norm = fsqrt.norm(p=2, dim=1, keepdim=True) + 1e-12
        fsqrt_l2 = fsqrt.div(norm)
        out = self.classifier(fsqrt_l2)
        return out


def cnnbilinear(num_classes=200):
    cnn1 = resnet18(pretrained=True)
    cnn1 = nn.Sequential(*list(cnn1.children())[:-1])
    cnn2 = resnet18(pretrained=True)
    cnn2 = nn.Sequential(*list(cnn2.children())[:-1])
    return Bilinear(cnn1, cnn2, num_classes)