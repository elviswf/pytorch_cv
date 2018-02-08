# -*- coding: utf-8 -*-
"""
@Time    : 2018/2/5 15:48
@Author  : Elvis

CUDA_VISIBLE_DEVICES=7 python mnist_bilinear.py

mnist_bilinear1
"""
import os
import argparse
import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.backends import cudnn
from models.basenet import bilinearNet
from utils.logger import progress_bar

# Learning rate parameters
BASE_LR = 0.001
NUM_CLASSES = 10  # set the number of classes in your dataset
DATA_DIR = "/home/elvis/code/data/mnist"
BATCH_SIZE = 32
IMAGE_SIZE = 28
MODEL_NAME = "mnist_bilinear1"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

parser = argparse.ArgumentParser(description='PyTorch mnist_bilinear1 Training')
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
    net = bilinearNet(num_classes=NUM_CLASSES)

# print(torch_summarize(net))
# print(net)
if USE_GPU:
    net.cuda()
    cudnn.benchmark = True

log = open("./log/%s.txt" % MODEL_NAME, 'a')
print("==> Preparing data...")
kwargs = {'num_workers': 2, 'pin_memory': True} if USE_GPU else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()


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

    log.write(str(epoch) + ' ' + str(correct / total) + ' ')


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

optim_params = list(net.bilinear1.parameters())
for param in optim_params:
    param.requires_grad = True

epoch1 = 8
# optimizer = optim.Adagrad(optim_params, lr=0.001, weight_decay=0.005)
optimizer = optim.Adam(optim_params, weight_decay=0.0005)
if start_epoch < epoch1:
    for epoch in range(start_epoch, epoch1):
        train(epoch, net, optimizer)
        test(epoch, net)
    start_epoch = epoch1

for param in net.parameters():
    param.requires_grad = True

optim_params = list(net.parameters())
# fc_params = list(map(id, net.fc2.parameters()))
# base_params = list(filter(lambda p: id(p) not in fc_params, net.parameters()))

optimizer = optim.SGD(optim_params, lr=0.0001, weight_decay=0.0005)
# optimizer = optim.Adagrad(optim_params, lr=0.001, weight_decay=0.0005)
for epoch in range(start_epoch, 200):
    train(epoch, net, optimizer)
    test(epoch, net)
log.close()
