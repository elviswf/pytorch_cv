# -*- coding: utf-8 -*-
"""
@Time    : 2018/1/25 14:04
@Author  : Elvis
"""
"""
 zsl_resnet.py
  
"""
import numpy as np
import random
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, orthogonal
from torch.autograd import Variable, Function
from torchvision.models import resnet18, resnet50, resnet101


class AttriCNN(nn.Module):
    def __init__(self, cnn, w_attr, num_attr=312, num_classes=150):
        super(AttriCNN, self).__init__()
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


def attrCNN_cubfull(num_attr=312, num_classes=200):
    cnn = resnet50(pretrained=True)
    w_attr = np.load("data/order_cub_attr.npy")
    w_attr = torch.FloatTensor(w_attr / 100.)  # 312 * 200
    return AttriWeightedCNN(cnn=cnn, w_attr=w_attr, num_attr=num_attr, num_classes=num_classes)


def attrCNN_awa2(num_attr=85, num_classes=40):
    cnn = resnet18(pretrained=True)
    w_attr = np.load("data/order_awa2_attr.npy")
    w_attr = w_attr[:num_classes, :]
    w_attr = torch.FloatTensor(w_attr / 100.)
    return AttriCNN(cnn=cnn, w_attr=w_attr, num_attr=num_attr, num_classes=num_classes)


class AttriWeightedCNN(nn.Module):
    def __init__(self, cnn, w_attr, num_attr=312, num_classes=150):
        super(AttriWeightedCNN, self).__init__()
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        self.feat_size = cnn.fc.in_features

        self.fc0 = nn.Linear(self.feat_size, num_attr)
        # orthogonal(self.fc0.weight)

        self.fc1 = nn.Sequential(
            nn.Linear(self.feat_size, num_attr),
            nn.Sigmoid(),
            nn.Dropout(0.5)
        )
        # for m in self.fc1:
        #     if hasattr(m, 'weight'):
        #         orthogonal(m.weight)
        self.fc2 = nn.Linear(num_attr, num_classes, bias=False)
        self.fc2.weight = nn.Parameter(w_attr, requires_grad=False)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.view(feat.shape[0], -1)
        # wt = self.fc0_drop(self.fc0(feat))
        wt = self.fc0(feat)
        attr = self.fc1(feat)
        xt = attr.mul(wt)
        attr_y = self.fc2(xt)  # (batch,   square sum root
        return attr_y, attr


class WARP(Function):
    '''
    autograd function of WARP loss
    '''
    @staticmethod
    def forward(ctx, input, target, max_num_trials=None):

        batch_size = target.size()[0]
        if max_num_trials is None:
            max_num_trials = target.size()[1] - 1

        positive_indices = torch.zeros(input.size()).cuda()
        negative_indices = torch.zeros(input.size()).cuda()
        L = torch.zeros(input.size()[0]).cuda()

        all_labels_idx = np.arange(target.size()[1])

        Y = float(target.size()[1])
        J = torch.nonzero(target)

        for i in range(batch_size):

            msk = np.ones(target.size()[1], dtype=bool)

            # Find the positive label for this example
            j = J[i, 1]
            positive_indices[i, j] = 1
            msk[j] = False

            # initialize the sample_score_margin
            sample_score_margin = -1
            num_trials = 0

            neg_labels_idx = all_labels_idx[msk]

            while sample_score_margin < 0 and num_trials < max_num_trials:
                # randomly sample a negative label
                neg_idx = random.sample(list(neg_labels_idx), 1)[0]
                msk[neg_idx] = False
                neg_labels_idx = all_labels_idx[msk]

                num_trials += 1
                # calculate the score margin
                sample_score_margin = 1 + input[i, neg_idx] - input[i, j]

            if sample_score_margin < 0:
                # checks if no violating examples have been found
                continue
            else:
                loss_weight = np.log(math.floor((Y - 1) / num_trials))
                L[i] = loss_weight
                negative_indices[i, neg_idx] = 1

        loss = L * (1 - torch.sum(positive_indices * input, dim=1) + torch.sum(negative_indices * input, dim=1))

        ctx.save_for_backward(input, target)
        ctx.L = L
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices

        return torch.sum(loss, dim=0, keepdim=True)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_variables
        L = Variable(torch.unsqueeze(ctx.L, 1), requires_grad=False)

        positive_indices = Variable(ctx.positive_indices, requires_grad=False)
        negative_indices = Variable(ctx.negative_indices, requires_grad=False)
        grad_input = grad_output * L * (negative_indices - positive_indices)

        return grad_input, None, None


class WARPLoss(nn.Module):
    def __init__(self, max_num_trials=None):
        super(WARPLoss, self).__init__()
        self.max_num_trials = max_num_trials

    def forward(self, input, target):
        return WARP.apply(input, target, self.max_num_trials)


def CNNw(num_classes=150):
    cnn = resnet101(pretrained=True)
    feat_size = cnn.fc.in_features
    cnn.fc = nn.Linear(feat_size, num_classes, bias=False)
    return cnn
