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
from torchvision.models.resnet import resnet18
from torch.nn.init import kaiming_normal


class SpatialTransformNet(nn.Module):
    def __init__(self, classifier):
        super(SpatialTransformNet, self).__init__()
        self.classifier = classifier

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(10, 10, kernel_size=5),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 24 * 24, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 24 * 24)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, theta

    def forward(self, x):
        # transform the input
        x, theta = self.stn(x)
        out = self.classifier(x)
        return out, theta


class STNLayer(nn.Module):
    def __init__(self, localization_net):
        super(STNLayer, self).__init__()
        self.localization_net = localization_net



def stnResnet(num_classes=200):
    model = resnet18(pretrained=True)
    num_features = model.fc.in_features
    print(num_features)
    model.fc = nn.Linear(num_features, num_classes)
    # return model
    return SpatialTransformNet(model)


class STN2Resnet(nn.Module):
    def __init__(self, num_classes=200):
        super(STN2Resnet, self).__init__()
        self.model = resnet18(pretrained=True)
        self.feature1 = nn.Sequential(*list(self.model.children())[:-1])
        import copy
        self.feature2 = copy.deepcopy(self.feature1)
        self.stn1 = SpatialTransformNet(self.feature1)
        self.stn2 = SpatialTransformNet(self.feature2)
        self.linear = nn.Linear(self.model.fc.in_features * 2, num_classes)
        kaiming_normal(self.linear.weight)

    def forward(self, x):
        # transform the input
        x1, theta1 = self.stn1(x)
        x2, theta2 = self.stn2(x)
        x1 = x1.view(-1, 512)
        x2 = x2.view(-1, 512)
        cat_x = torch.cat((x1, x2), 1)
        theta1 = theta1.view(-1, 6)
        theta2 = theta2.view(-1, 6)
        theta = torch.cat((theta1, theta2), 1)
        out = self.linear(cat_x)
        return out, theta,


class STN4Resnet(nn.Module):
    def __init__(self, num_classes=200):
        super(STN4Resnet, self).__init__()
        self.model = resnet18(pretrained=True)
        self.feature1 = nn.Sequential(*list(self.model.children())[:-1])
        import copy
        self.feature2 = copy.deepcopy(self.feature1)
        self.feature3 = copy.deepcopy(self.feature1)
        self.feature4 = copy.deepcopy(self.feature1)
        self.stn1 = SpatialTransformNet(self.feature1)
        self.stn2 = SpatialTransformNet(self.feature2)
        self.stn3 = SpatialTransformNet(self.feature3)
        self.stn4 = SpatialTransformNet(self.feature4)
        self.linear = nn.Linear(self.model.fc.in_features * 4, num_classes)

    def forward(self, x):
        # transform the input
        x1, theta1 = self.stn1(x)
        x2, theta2 = self.stn2(x)
        x3, theta3 = self.stn3(x)
        x4, theta4 = self.stn4(x)
        x1 = x1.view(-1, 512)
        x2 = x2.view(-1, 512)
        x3 = x3.view(-1, 512)
        x4 = x4.view(-1, 512)
        cat_x = torch.cat((x1, x2, x3, x4), 1)
        theta1 = theta1.view(-1, 6)
        theta2 = theta2.view(-1, 6)
        theta3 = theta3.view(-1, 6)
        theta4 = theta4.view(-1, 6)
        theta = torch.cat((theta1, theta2, theta3, theta4), 1)
        out = self.linear(cat_x)
        return out, theta