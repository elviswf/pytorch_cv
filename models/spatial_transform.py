# -*- coding: utf-8 -*-
"""
@Time    : 2017/11/20 14:32
@Author  : Elvis
"""
"""
 spatial_transform.py
  
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.resnet import resnet18
from torch.nn.init import kaiming_normal
import copy


def rangeScale(x):
    # y = x.clamp(-0.2, 0.2)
    y = torch.exp(x) + 1
    return 0.5 - torch.reciprocal(y)


def squash(x):
    y = torch.norm(x)


class SpatialTransformNet(nn.Module):
    def __init__(self, cnn, num_classes=200):
        super(SpatialTransformNet, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            *list(cnn.children())[:-1],
            nn.Conv2d(512, 128, kernel_size=1)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 14 * 14, 128),
            nn.Linear(128, 4)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[1].weight.data.fill_(0)
        self.fc_loc[1].bias.data = torch.FloatTensor([-0.1, 0, 0, 0.1])

        self.cnn1 = copy.deepcopy(cnn)
        self.cnn2 = copy.deepcopy(cnn)
        self.linear = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )
        kaiming_normal(self.linear[1].weight)

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(x.size(0), -1)
        theta = self.fc_loc(xs)
        # theta = rangeScale(theta)
        theta1 = Variable(torch.zeros(x.size(0), 2, 3)).cuda()
        diag = torch.diag(torch.ones(2)*0.5)
        theta1[:, :, -1] = theta[:, :2]
        theta1[:, :, :2] = Variable(diag.expand(x.size(0), 2, 2), requires_grad=False)
        theta1 = theta1.contiguous()
        theta2 = Variable(torch.zeros(x.size(0), 2, 3)).cuda()
        theta2[:, :, -1] = theta[:, 2:]
        theta2[:, :, :2] = Variable(diag.expand(x.size(0), 2, 2), requires_grad=False)
        theta2 = theta2.contiguous()
        x_size = x.size()
        sample_size = torch.Size([x_size[0], x_size[1], 224, 224])

        grid1 = F.affine_grid(theta1, sample_size)
        x1 = F.grid_sample(x, grid1)
        grid2 = F.affine_grid(theta2, sample_size)
        x2 = F.grid_sample(x, grid2)
        return x1, x2, theta

    def forward(self, x):
        # transform the input
        x1, x2, theta = self.stn(x)
        out1 = self.cnn1(x1)
        out2 = self.cnn2(x2)
        cat_x = torch.cat((out1, out2), 1)
        cat_x = cat_x.view(x.size(0), -1)
        out = self.linear(cat_x)
        return out, theta, x1


def stnResnet(num_classes=200):
    model = resnet18(pretrained=True)
    cnn = nn.Sequential(*list(model.children())[:-1])
    # return model
    return SpatialTransformNet(cnn, num_classes)

