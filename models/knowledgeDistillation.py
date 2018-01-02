# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/13 14:59
@Author  : Elvis
"""
"""
 loss.py
  
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models.resnet import resnet18

temperature = 2.0


class KnowledgeDistillationNet(nn.Module):
    def __init__(self, num_classes=10):
        super(KnowledgeDistillationNet, self).__init__()
        pretrain = resnet18(pretrained=True)
        self.base = nn.Sequential(*list(pretrain.children())[:-1])
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        feature = self.base(x)
        logits = self.linear(feature)
        prob = F.softmax(logits)
        prob_t = F.softmax(logits / temperature)
        return prob, prob_t


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_true, y_soft, y_pred, y_pred_t):
        # convert logits to soft targets
        y_soft = F.softmax(y_soft / temperature)
        return F.cross_entropy(y_true, y_pred) + self.alpha * F.cross_entropy(y_soft, y_pred_t)
