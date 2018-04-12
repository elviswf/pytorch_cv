# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/4 15:42
@Author  : Elvis

 awa2.py
watch --color -n1 gpustat -cpu
CUDA_VISIBLE_DEVICES=1 python awa2_attr1.py

zsl_resnet18_fc00_awa2 : Sigmoid + dropout 0.5 Acc: 92.320% (5614/6081)   ZSL Acc: 24.939% (1742/6985)
Acc: 72.684% (5077/6985)
"""
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.autograd import Variable
import os
import argparse
from data.data_loader import DataLoader
from models.zsl_resnet import attrCNNg_awa2, RegLoss
from utils.logger import progress_bar
from zeroshot.awa2_test import zsl_test, gzsl_test0
import copy
# from models.focalLoss import FocalLoss
# from utils.param_count import torch_summarize, lr_scheduler
# import pickle

# Learning rate parameters
BASE_LR = 0.01
NUM_CLASSES = 50  # set the number of classes in your dataset
NUM_ATTR = 85
DATA_DIR = "/home/elvis/data/attribute/AwA/Animals_with_Attributes2/zsl/trainval"
BATCH_SIZE = 128
IMAGE_SIZE = 224
gamma = 5.
lamda2 = 0.0
MODEL_NAME = "gzsl_awa2_g5_la0"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

parser = argparse.ArgumentParser(description='PyTorch Training：%s' % MODEL_NAME)
parser.add_argument('--lr', default=BASE_LR, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--data', default=DATA_DIR, type=str, help='file path of the dataset')
args = parser.parse_args()

best_acc = 0.
start_epoch = 0
print("Model: " + MODEL_NAME)
if args.resume:
    print("==> Resuming from checkpoint...")
    MODEL_SAVE_FILE = "gzsl_awa2_epoch6acc55.pth"
    checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    optimizer = checkpoint["optimizer"]
else:
    print("==> Building model...")
    net = attrCNNg_awa2(num_attr=NUM_ATTR, num_classes=NUM_CLASSES)

# print(torch_summarize(net))
# print(net)
if USE_GPU:
    net.cuda()
    # net = torch.nn.DataParallel(net.module, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

log = open("./log/" + MODEL_NAME + '_awa2.txt', 'a')
print("==> Preparing data...")
data_loader = DataLoader(data_dir=args.data, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
inputs, classes = next(iter(data_loader.load_data()))
# out = torchvision.utils.make_grid(inputs)
# data_loader.show_image(out, title=[data_loader.data_classes[c] for c in classes])
train_loader = data_loader.load_data(data_set='train')
test_loader = data_loader.load_data(data_set='val')
# criterion = nn.CrossEntropyLoss()
criterion = RegLoss(lamda2, superclass="awa2")
# criterion = FocalLoss(class_num=NUM_CLASSES, gamma=0)


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

        out, attr = net(inputs)
        loss = criterion(out, targets, attr)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(train_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    log.write(str(epoch) + ' ' + str(correct / total) + ' ')


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, attr = net(inputs)
        loss = criterion(out, targets, attr)

        test_loss = loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        acc = 100. * correct / total
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), acc, correct, total))

    log.write(str(correct / total) + ' ' + str(test_loss) + ' ')

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


epoch1 = 4
# optimizer = optim.Adagrad(optim_params, lr=0.001, weight_decay=0.005)
if start_epoch < epoch1:
    for param in net.parameters():
        param.requires_grad = False
    # optim_params = list(net.fc0.parameters()) + list(net.fc1.parameters())
    optim_params = list(net.fc1.parameters())
    for param in optim_params:
        param.requires_grad = True
    optimizer = optim.Adam(optim_params, weight_decay=0.0005)
    for epoch in range(start_epoch, epoch1):
        train(epoch, net, optimizer)
        test(epoch, net)
        gzsl_test0(epoch, net, optimizer, log, gamma=gamma)
        net1 = copy.deepcopy(net)
        zsl_test(epoch, net1, optimizer, log)
        del net1
        log.write("\n")
        log.flush()
    start_epoch = epoch1

fc_params = list(map(id, net.fc2.parameters()))
base_params = list(filter(lambda p: id(p) not in fc_params, net.parameters()))

for param in base_params:
    param.requires_grad = True

optimizer = optim.Adagrad(base_params, lr=0.001, weight_decay=0.0005)

for epoch in range(start_epoch, 100):
    train(epoch, net, optimizer)
    test(epoch, net)
    gzsl_test0(epoch, net, optimizer, log, gamma=gamma)
    net1 = copy.deepcopy(net)
    zsl_test(epoch, net1, optimizer, log)
    del net1
    log.write("\n")
    log.flush()
log.close()

