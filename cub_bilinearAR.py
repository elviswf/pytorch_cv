# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/4 15:42
@Author  : Elvis

 cub.py
watch --color -n1 gpustat -cpu
CUDA_VISIBLE_DEVICES=4 python cub_attr1.py

resnet50_binearAR1 : resnetAR  act Acc: 79.997% (4635/5794)  Acc: 68.260% (3955/5794)
resnet50_binearAR2 : resnet2AR
resnet50_binearAR3 : resnetAR + relu 67%   no act:  Acc: 64.446% (3734/5794)
resnet50_binearAR4: NeWnet train not easy   Acc: 79.789% (4623/5794)

resnet18_binearAR1: resnet18ARfc   Acc: 74.698% (4328/5794)
resnet18_binearAR2: resnet18ARfc   bn   Acc: 74.629% (4324/5794)
resnet18_binearAR3: resnet18ARfc + sigmoid  no better Acc: 74.784% (4333/5794)
"""
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.autograd import Variable
import os
import argparse
from data.data_loader import DataLoader
# from models.attr_resnet import attrCNN, WARPLoss
from models.bilinearAR import resnetAR, resnet18AR, resnet18ARfc
from utils.logger import progress_bar

# from utils.param_count import torch_summarize, lr_scheduler
# import pickle

# Learning rate parameters
BASE_LR = 0.01
NUM_CLASSES = 200  # set the number of classes in your dataset
NUM_ATTR = 312
DATA_DIR = "/home/elvis/code/data/cub200"
BATCH_SIZE = 32
IMAGE_SIZE = 224
MODEL_NAME = "resnet18_binearAR3"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

parser = argparse.ArgumentParser(description='PyTorch resnet50_binearAR2 Training')
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
    net = resnet18ARfc(num_classes=200)

# print(torch_summarize(net))
# print(net)
log = open("./log/" + MODEL_NAME + '_cub.txt', 'a')
log.write(str(net.__repr__))

if USE_GPU:
    net.cuda()
    # net = torch.nn.DataParallel(net.module, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

print("==> Preparing data...")
data_loader = DataLoader(data_dir=args.data, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
inputs, classes = next(iter(data_loader.load_data()))
# out = torchvision.utils.make_grid(inputs)
# data_loader.show_image(out, title=[data_loader.data_classes[c] for c in classes])
train_loader = data_loader.load_data(data_set='train')
test_loader = data_loader.load_data(data_set='val')
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

optim_params = list(net.bilinear1.parameters()) + list(net.fc.parameters())

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
# optimizer = optim.SGD(net.parameters(), lr=0.0001, weight_decay=0.0005)
optimizer = optim.Adagrad(optim_params, lr=0.001, weight_decay=0.0005)
for epoch in range(start_epoch, 500):
    train(epoch, net, optimizer)
    test(epoch, net)
log.close()
