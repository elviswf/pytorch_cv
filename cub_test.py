# -*- coding: utf-8 -*-
"""
@Time    : 2018/1/25 16:06
@Author  : Elvis
"""
"""
 cub_test.py
  
"""
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision
import os
import argparse
from data.data_loader import DataLoader
from utils.logger import progress_bar
# from utils.param_count import torch_summarize, lr_scheduler
# import pickle

# Learning rate parameters
BASE_LR = 0.01
NUM_CLASSES = 200  # set the number of classes in your dataset
NUM_ATTR = 312
DATA_DIR = "/home/elvis/data/attribute/CUB_200_2011/zsl/zsl_test"
BATCH_SIZE = 32
IMAGE_SIZE = 224
MODEL_NAME = "zsl_resnet18_attr1_fc"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

parser = argparse.ArgumentParser(description='PyTorch zsl_resnet18_attr1 Training')
parser.add_argument('--lr', default=BASE_LR, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--data', default=DATA_DIR, type=str, help='file path of the dataset')
args = parser.parse_args()

best_acc = 0.
start_epoch = 0
print("Model: " + MODEL_NAME)
print("==> Resuming from checkpoint...")
checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
net = checkpoint["net"]
best_acc = checkpoint["acc"]
start_epoch = checkpoint["epoch"]
optimizer = checkpoint["optimizer"]

w_attr = np.load("data/w_attr.npy")
# w_attr_sum = np.sum(w_attr, 0)
# w_attr = w_attr/w_attr_sum
# w_attr[:, 0].sum()

w_attr = torch.FloatTensor(w_attr)  # 150 * 312

net.fc2 = nn.Linear(NUM_ATTR, NUM_CLASSES, bias=False)
net.fc2 = nn.Parameter(w_attr, requires_grad=False)
# print(torch_summarize(net))
# print(net)
if USE_GPU:
    net.cuda()
    # net = torch.nn.DataParallel(net.module, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

print("==> Preparing data...")
data_loader = DataLoader(data_dir=args.data, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
inputs, classes = next(iter(data_loader.load_data()))
out = torchvision.utils.make_grid(inputs)
data_loader.show_image(out, title=[data_loader.data_classes[c] for c in classes])
train_loader = data_loader.load_data(data_set='train')
test_loader = data_loader.load_data(data_set='val')
criterion = nn.CrossEntropyLoss()


def one_hot_emb(y, depth=NUM_CLASSES):
    y = y.view((-1, 1))
    one_hot = torch.FloatTensor(y.size(0), depth).zero_()
    one_hot.scatter_(1, y, 1)
    return one_hot


def train(epoch, net, optimizer):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # optimizer = lr_scheduler(optimizer, epoch, init_lr=0.002, decay_epoch=start_epoch)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()

        out = net(inputs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(train_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out = net(inputs)
        loss = criterion(out, targets)

        test_loss = loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        acc = 100. * correct / total
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), acc, correct, total))

    acc = 100. * correct / total
    if epoch > 9 and acc > best_acc:
        print("Saving checkpoint")
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer
        }
        if not os.path.isdir("checkpoints"):
            os.mkdir('checkpoints')
        torch.save(state, "./checkpoints/" + MODEL_SAVE_FILE)
        best_acc = acc


epoch = 0
test(epoch, net)
