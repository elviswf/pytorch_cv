# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/4 15:42
@Author  : Elvis

 cub.py
watch --color -n1 gpustat -cpu
CUDA_VISIBLE_DEVICES=4 python cub_attr1.py

cifar_base1 :   Acc: 65.340% (6534/10000)  +dropout
cifar_base2 : Acc: 82.320% (8232/10000)
cifar_base3 :

"""
import torch
from torch import optim
from torch.backends import cudnn
from torch import nn
import torchvision
from torchvision import transforms
import os
import argparse
from models.basenet import resnetBase
from utils.logger import progress_bar
from torch.autograd import Variable

# Learning rate parameters
BASE_LR = 0.01
NUM_CLASSES = 100  # set the number of classes in your dataset
BATCH_SIZE = 64
IMAGE_SIZE = 224
DATA_DIR = "/home/elvis/code/data/cifar"
MODEL_NAME = "cifar_base2"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

parser = argparse.ArgumentParser(description='PyTorch cifar_base Training')
parser.add_argument('--lr', default=BASE_LR, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
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
    net = resnetBase(num_classes=NUM_CLASSES)

print("==> Preparing data...")
transform_train = transforms.Compose([
    # transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transforms_test = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=False, transform=transforms_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# print(torch_summarize(net))
# print(net)
if USE_GPU:
    net.cuda()
    # net = torch.nn.DataParallel(net.module, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

log = open("./log/" + MODEL_NAME + '_cifar100.txt', 'a')
print("==> Preparing data...")
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

optim_params = list(net.fc.parameters())
for param in optim_params:
    param.requires_grad = True

epoch1 = 12
# optimizer = optim.Adagrad(optim_params, lr=0.001, weight_decay=0.005)
optimizer = optim.Adam(optim_params, weight_decay=0.0005)
if start_epoch < epoch1:
    for epoch in range(start_epoch, epoch1):
        train(epoch, net, optimizer)
        test(epoch, net)
    start_epoch = epoch1

optim_params = list(net.parameters())
for param in optim_params:
    param.requires_grad = True

# fc_params = list(map(id, net.fc2.parameters()))
# base_params = list(filter(lambda p: id(p) not in fc_params, net.parameters()))

# optimizer = optim.SGD(optim_params, lr=0.0001, weight_decay=0.0005)
optimizer = optim.Adagrad(optim_params, lr=0.001, weight_decay=0.0005)
for epoch in range(start_epoch, 200):
    train(epoch, net, optimizer)
    test(epoch, net)
log.close()
