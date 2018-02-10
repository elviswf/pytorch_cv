# -*- coding: utf-8 -*-
"""
@Time    : 2018/1/25 14:04
@Author  : Elvis
 zsl_resnet.py
for m in self.fc1:
    if hasattr(m, 'weight'):
        orthogonal(m.weight)
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet18, resnet50, resnet101


# from torch.nn.init import kaiming_normal, orthogonal


class AttriCNN(nn.Module):
    def __init__(self, cnn, w_attr, num_attr=312, num_classes=150):
        super(AttriCNN, self).__init__()
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        self.feat_size = cnn.fc.in_features

        self.fc0 = nn.Sequential(
            nn.Linear(num_attr, num_attr),
            # nn.Sigmoid(),
        )
        self.fc1 = nn.Sequential(
            # nn.Linear(self.feat_size, 32),
            # nn.Linear(32, num_attr),
            nn.Linear(self.feat_size, num_attr),
            nn.Dropout(0.5),
            # nn.Tanh(),
            nn.Sigmoid(),
        )

        self.fc2 = nn.Linear(num_attr, num_classes, bias=False)
        self.fc2.weight = nn.Parameter(w_attr, requires_grad=False)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.view(feat.shape[0], -1)
        wt = self.fc1(feat)
        attr = self.fc0(wt)
        attr_y = self.fc2(attr)  # xt (batch,   square sum root
        return attr_y, attr


def attrCNN(num_attr=312, num_classes=150):
    cnn = resnet50(pretrained=True)
    w_attr = np.load("data/order_cub_attr.npy")
    # w_attr_sum = np.sum(w_attr, 0)
    # w_attr = w_attr/w_attr_sum
    # w_attr[:, 0].sum()
    w_attr = w_attr[:num_classes, :] / 100.
    w_attr = torch.FloatTensor(w_attr)  # 312 * 150
    # (torch.ones((1, 2)).mm(torch.ones((2, 3))))
    return AttriCNN(cnn=cnn, w_attr=w_attr, num_attr=num_attr, num_classes=num_classes)


def attrWeightedCNN(num_attr=312, num_classes=150):
    cnn = resnet50(pretrained=True)
    w_attr = np.load("data/order_cub_attr.npy")
    w_attr = w_attr[:num_classes, :] / 100.
    w_attr = torch.FloatTensor(w_attr)  # 312 * 150
    return AttriWeightedCNN(cnn=cnn, w_attr=w_attr, num_attr=num_attr, num_classes=num_classes)


def attrWCNNg(num_attr=312, num_classes=200):
    cnn = resnet50(pretrained=True)
    w_attr = np.load("data/order_cub_attr.npy")
    w_attr = w_attr / 100.
    w_attr = torch.FloatTensor(w_attr)  # 312 * 150
    return AttriWeightedCNN(cnn=cnn, w_attr=w_attr, num_attr=num_attr, num_classes=num_classes)


def attrCNN_cubfull(num_attr=312, num_classes=200):
    cnn = resnet50(pretrained=True)
    w_attr = np.load("data/order_cub_attr.npy")
    w_attr = torch.FloatTensor(w_attr / 100.)  # 312 * 200
    return AttriWeightedCNN(cnn=cnn, w_attr=w_attr, num_attr=num_attr, num_classes=num_classes)


def attrCNN_awa2(num_attr=85, num_classes=50):
    cnn = resnet18(pretrained=True)
    w_attr = np.load("data/order_awa2_attr.npy")
    # w_attr = w_attr[:num_classes, :]
    w_attr = torch.FloatTensor(w_attr / 100.)
    return AttriWeightedCNN(cnn=cnn, w_attr=w_attr, num_attr=num_attr, num_classes=num_classes)


class AttriWeightedCNN(nn.Module):
    def __init__(self, cnn, w_attr, num_attr=312, num_classes=150):
        super(AttriWeightedCNN, self).__init__()
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        self.feat_size = cnn.fc.in_features

        self.fc0 = nn.Sequential(
            nn.Linear(self.feat_size, num_attr),
            # nn.Sigmoid(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.feat_size, num_attr),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            # nn.Tanh(),
            # nn.Linear(self.feat_size, 32),
            # nn.Linear(32, num_attr),
        )

        self.fc2 = nn.Linear(num_attr, num_classes, bias=False)
        self.fc2.weight = nn.Parameter(w_attr, requires_grad=False)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.view(feat.shape[0], -1)
        attr = self.fc0(feat)
        wt = self.fc1(feat)
        xt = wt.mul(attr)
        attr_y = self.fc2(xt)  # xt (batch,   square sum root
        return attr_y, attr


def CNNw(num_classes=150):
    cnn = resnet101(pretrained=True)
    feat_size = cnn.fc.in_features
    cnn.fc = nn.Linear(feat_size, num_classes, bias=False)
    return cnn


class DeepRIS(nn.Module):
    def __init__(self, cnn, w_attr, num_attr=312, num_classes=150):
        super(DeepRIS, self).__init__()
        self.cnn = cnn
        feat_size = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(feat_size, num_attr),
            nn.Sigmoid(),
            nn.Dropout(0.4)
        )
        self.fc2 = nn.Linear(num_attr, num_classes, bias=False)
        self.fc2.weight = nn.Parameter(w_attr, requires_grad=False)
        # for m in self.cnn.fc:
        #     if hasattr(m, 'weight'):
        #         orthogonal(m.weight)

    def forward(self, x):
        attr = self.cnn(x)
        attr_y = self.fc2(attr)  # (batch,   square sum root
        return attr_y, attr


def soft_celoss(logit, prob):
    """ Cross-entropy function"""
    soft_logit = F.log_softmax(logit, dim=1)
    loss = torch.sum(prob * soft_logit, 1)
    return loss


def soft_loss(out, targets):
    """Compute the total loss"""
    ws = np.load("data/cub_ws_18.npy")
    ws = torch.FloatTensor(ws).cuda()
    targets_data = targets.data
    targets_data = targets_data.type(torch.cuda.LongTensor)
    soft_target = ws[targets_data]
    soft_target = Variable(soft_target, requires_grad=False).cuda()
    soft_ce = - torch.mean(soft_celoss(out, soft_target))

    ce = F.cross_entropy(out, targets)
    alpha = 0.5
    loss = alpha * ce + (1. - alpha) * soft_ce
    return loss
