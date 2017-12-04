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
from torchvision.models import resnet50


class EmbResNet(nn.Module):
    def __init__(self, num_classes=200):
        super(EmbResNet, self).__init__()
        self.pretrain = resnet50(pretrained=True)
        self.feature = nn.Sequential(*list(self.pretrain.children())[:-1])

        self.linear1 = nn.Linear(2048, num_classes)
        self.linear2 = nn.Linear(2048, num_classes)
        self.emb = nn.Embedding(num_classes, num_classes)
        self.emb.weight = nn.Parameter(torch.eye(num_classes))

    def forward(self, x, targets, epoch, batch_idx):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        out2 = self.linear2(out.detach())
        tar = self.emb(targets).cuda()
        return out1, out2, tar, self.emb.weight
