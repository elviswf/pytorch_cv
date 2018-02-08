# -*- coding: utf-8 -*-
"""
@Time    : 2018/1/25 16:06
@Author  : Elvis
"""
"""
CUDA_VISIBLE_DEVICES=0 python cub_test.py
  
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from data.data_loader import DataLoader
from utils.logger import progress_bar


def zsl_test(epoch, net, optimizer):
    NUM_CLASSES = 200  # set the number of classes in your dataset
    NUM_ATTR = 312
    DATA_DIR = "/home/elvis/data/attribute/CUB_200_2011/zsl/zsl_test"
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    best_acc = 67
    USE_GPU = torch.cuda.is_available()
    order_cub_attr = np.load("data/order_cub_attr.npy")
    # w_attr_sum = np.sum(w_attr, 0)
    # w_attr = w_attr/w_attr_sum
    # w_attr[:, 0].sum()
    order_cub_attr = order_cub_attr[150:, :]/ 100.
    order_cub_attr = torch.FloatTensor(order_cub_attr).cuda()  # 50 * 312
    net.fc2 = nn.Linear(NUM_ATTR, NUM_CLASSES, bias=False)
    net.fc2.weight = nn.Parameter(order_cub_attr, requires_grad=False)
    # print(torch_summarize(net))
    # print(net)
    net.cuda()
    data_loader = DataLoader(data_dir=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    # train_loader = data_loader.load_data(data_set='train')
    test_loader = data_loader.load_data(data_set='val')
    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, attr = net(inputs)
        loss = criterion(out, targets)

        test_loss = loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        acc = 100. * correct / total
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), acc, correct, total))

    acc = 100. * correct / total
    if acc > best_acc:
        MODEL_SAVE_FILE = "zsl_resnet50_cub_epoch%dacc%d.pth" % (epoch, int(acc))
        print(MODEL_SAVE_FILE)
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer
        }
        torch.save(state, "./checkpoints/" + MODEL_SAVE_FILE)


def gzsl_test(epoch, net, optimizer):
    NUM_CLASSES = 200  # set the number of classes in your dataset
    NUM_ATTR = 312
    DATA_DIR = "/home/elvis/data/attribute/CUB_200_2011/zsl/gzsl_test"
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    best_acc = 63
    USE_GPU = torch.cuda.is_available()
    order_cub_attr = np.load("data/order_cub_attr.npy")
    # order_cub_attr = order_cub_attr[150:, :]
    order_cub_attr = torch.FloatTensor(order_cub_attr / 100.).cuda()  # 50 * 312
    net.fc2 = nn.Linear(NUM_ATTR, NUM_CLASSES, bias=False)
    net.fc2.weight = nn.Parameter(order_cub_attr, requires_grad=False)
    net.cuda()
    data_loader = DataLoader(data_dir=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    # train_loader = data_loader.load_data(data_set='train')
    test_loader = data_loader.load_data(data_set='val')
    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, attr = net(inputs)
        loss = criterion(out, targets)

        test_loss = loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        acc = 100. * correct / total
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), acc, correct, total))

    acc = 100. * correct / total
    if acc > best_acc:
        MODEL_SAVE_FILE = "gzsl_resnet50_cub_epoch%dacc%d.pth" % (epoch, int(acc))
        print(MODEL_SAVE_FILE)
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer
        }
        torch.save(state, "./checkpoints/" + MODEL_SAVE_FILE)


def gzsl_test0(epoch, net, optimizer):
    NUM_CLASSES = 200  # set the number of classes in your dataset
    NUM_ATTR = 312
    DATA_DIR = "/home/elvis/code/data/cub200"
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    best_acc = 60
    USE_GPU = torch.cuda.is_available()
    order_cub_attr = np.load("data/order_cub_attr.npy")
    # order_cub_attr = order_cub_attr[150:, :]
    order_cub_attr = torch.FloatTensor(order_cub_attr / 100.).cuda()  # 50 * 312
    net.fc2 = nn.Linear(NUM_ATTR, NUM_CLASSES, bias=False)
    net.fc2.weight = nn.Parameter(order_cub_attr, requires_grad=False)
    net.cuda()
    data_loader = DataLoader(data_dir=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    train_loader = data_loader.load_data(data_set='train')
    test_loader = data_loader.load_data(data_set='val')
    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, attr = net(inputs)
        loss = criterion(out, targets)

        test_loss = loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        acc = 100. * correct / total
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), acc, correct, total))

    acc = 100. * correct / total
    if acc > best_acc:
        MODEL_SAVE_FILE = "gzsl_resnet50_cub_epoch%dacc%d.pth" % (epoch, int(acc))
        print(MODEL_SAVE_FILE)
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer
        }
        torch.save(state, "./checkpoints/" + MODEL_SAVE_FILE)

