# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/23 15:39
@Author  : Elvis
"""
"""
 spn.py
  
"""
from . import spatialpooling
import torch.nn as nn
import torch
import torchvision.models as models
from spn.modules import SoftProposal


class SPNetWSL(nn.Module):
    def __init__(self, model, num_classes, num_maps, pooling):
        super(SPNetWSL, self).__init__()

        self.features = nn.Sequential(*list(model.children())[:-2])
        self.spatial_pooling = pooling

        # classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_maps, num_classes)
        )

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)
        x = self.spatial_pooling(x)
        x = x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_config_optim(self, lr):
        return [{'params': self.features.parameters(), 'lr': lr},
                {'params': self.spatial_pooling.parameters()},
                {'params': self.classifier.parameters()}]


def resnet18_sp(num_classes, pretrained=True, num_maps=1024):
    model = models.resnet18(pretrained)
    num_features = 512
    pooling = nn.Sequential()
    pooling.add_module('adconv',
                       nn.Conv2d(num_features, num_maps, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
    pooling.add_module('maps', nn.ReLU())
    pooling.add_module('sp', SoftProposal())
    pooling.add_module('sum', spatialpooling.SpatialSumOverMap())
    return SPNetWSL(model, num_classes, num_maps, pooling)


def resnet50_sp(num_classes, pretrained=True, num_maps=1024):
    model = models.resnet50(pretrained)
    num_features = 2048
    pooling = nn.Sequential()
    pooling.add_module('adconv',
                       nn.Conv2d(num_features, num_maps, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
    pooling.add_module('maps', nn.ReLU())
    pooling.add_module('sp', SoftProposal())
    pooling.add_module('sum', spatialpooling.SpatialSumOverMap())
    return SPNetWSL(model, num_classes, num_maps, pooling)
