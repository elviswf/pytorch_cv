# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/4 15:42
@Author  : Elvis

 cub.py
watch --color -n1 gpustat -cpu
CUDA_VISIBLE_DEVICES=2 python cub_attr.py

zsl_cub_resnet18_WARPLoss: Sigmoid + dropout 0.5 weight_decay=0.0005
"""
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision
import os
import argparse
from data.data_loader import DataLoader
from models.zsl_resnet import attrCNN, WARPLoss
from utils.logger import progress_bar
# from utils.param_count import torch_summarize, lr_scheduler
# import pickle

# Learning rate parameters
BASE_LR = 0.01
NUM_CLASSES = 150  # set the number of classes in your dataset
NUM_ATTR = 312
DATA_DIR = "/home/elvis/data/attribute/CUB_200_2011/zsl/trainval"
BATCH_SIZE = 32
IMAGE_SIZE = 224
MODEL_NAME = "zsl_cub_resnet18_WARPLoss"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

parser = argparse.ArgumentParser(description='PyTorch zsl_resnet18_attr Training')
parser.add_argument('--lr', default=BASE_LR, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--data', default=DATA_DIR, type=str, help='file path of the dataset')
args = parser.parse_args()

best_acc = 0.
start_epoch = 0
print("Model: " + MODEL_NAME)
if args.resume:
    print("==> Resuming from checkpoint...")
    checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    optimizer = checkpoint["optimizer"]
else:
    print("==> Building model...")
    net = attrCNN(num_attr=NUM_ATTR, num_classes=NUM_CLASSES)

# optimizer = optim.Adam(net.parameters())
# optimizer = optim.SGD(net.get_config_optim(BASE_LR / 10.),
#                       lr=BASE_LR,
#                       momentum=0.9,
#                       weight_decay=0.0005)
# print(torch_summarize(net))
# print(net)
if USE_GPU:
    net.cuda()
    # net = torch.nn.DataParallel(net.module, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

log = open("./log/" + MODEL_NAME + '_cub.txt', 'a')
print("==> Preparing data...")
data_loader = DataLoader(data_dir=args.data, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
inputs, classes = next(iter(data_loader.load_data()))
# out = torchvision.utils.make_grid(inputs)
# data_loader.show_image(out, title=[data_loader.data_classes[c] for c in classes])
train_loader = data_loader.load_data(data_set='train')
test_loader = data_loader.load_data(data_set='val')
criterion = WARPLoss()


# def one_hot_emb(batch, depth=NUM_CLASSES):
#     emb = nn.Embedding(depth, depth)
#     emb.weight.data = torch.eye(depth)
#     return emb(batch).data
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
        targets_emb = one_hot_emb(targets)
        if USE_GPU:
            inputs, targets_emb = inputs.cuda(), targets_emb.cuda()
        inputs, targets_emb = Variable(inputs), Variable(targets_emb)
        optimizer.zero_grad()

        out = net(inputs)
        loss = criterion(out, targets_emb)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets_emb.size(0)
        correct += predicted.cpu().eq(targets).sum()

        progress_bar(batch_idx, len(train_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    log.write(str(epoch) + ' ' + str(correct / total) + ' ')


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        targets_emb = one_hot_emb(targets)
        if USE_GPU:
            inputs, targets_emb = inputs.cuda(), targets_emb.cuda()
        inputs, targets_emb = Variable(inputs, volatile=True), Variable(targets_emb)
        out = net(inputs)
        loss = criterion(out, targets_emb)

        test_loss = loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets_emb.size(0)
        correct += predicted.cpu().eq(targets).sum()

        acc = 100. * correct / total
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), acc, correct, total))
    log.write(str(correct / total) + ' ' + str(test_loss) + '\n')
    log.flush()

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


for param in net.parameters():
    param.requires_grad = False

optim_params = list(net.cnn.parameters())
for param in optim_params:
    param.requires_grad = True

# optimizer = optim.Adagrad(optim_params, lr=0.001, weight_decay=0.0005)
from zeroshot.cub_test import zsl_test
import copy
for epoch in range(start_epoch, 300):
    train(epoch, net, optimizer)
    test(epoch, net)
    net1 = copy.deepcopy(net)
    zsl_test(epoch, net1, optimizer)

# for epoch in range(start_epoch, 100):
#     train(epoch, net, optimizer)
#     test(epoch, net)
log.close()

