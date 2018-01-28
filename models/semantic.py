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


class SemanticNet(nn.Module):
    def __init__(self, cnn, num_classes=200, embedding_dim=512):
        super(SemanticNet, self).__init__()
        self.cnn = cnn
        # self.fc0 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes)
        # )
        self.linear1 = nn.Linear(num_classes, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, 512)
        self.y2v = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
        )

    def forward(self, x, y):
        v_x = self.cnn(x)
        v_x = v_x.squeeze()
        v_y = self.y2v(y)
        return v_x, v_y


def semanticResnet(num_classes=200):
    cnn = resnet18(pretrained=True)
    cnn = nn.Sequential(*list(cnn.children())[:-1])
    return SemanticNet(cnn, num_classes)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, inputs, targets):
        diff = inputs - targets
        loss = torch.mean(torch.sum(diff.mul(diff), 1))
        return loss


def predict_y(vx, vy):
    batch_size = vx.size(0)
    num_points, points_dim = vy.size()
    vx = vx.view(batch_size, 1, -1)
    vx = vx.repeat(1, num_points, 1)
    vy = vy.view((1, num_points, points_dim))
    vy = vy.repeat(batch_size, 1, 1)
    diff = vx - vy
    dist = torch.sum(diff.mul(diff), 2)
    values, indices = dist.min(dim=1)
    return indices
