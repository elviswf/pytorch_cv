# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/4 15:42
@Author  : Elvis

 awa2.py
watch --color -n1 gpustat -cpu
CUDA_VISIBLE_DEVICES=1 python awa2_attr1.py

zsl_resnet18_fc00 : Sigmoid + dropout 0.5

"""
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.autograd import Variable
import os
import argparse
from data.data_loader import DataLoader
from models.zsl_resnet import attrCNN_awa2, WARPLoss
from utils.logger import progress_bar
# from utils.param_count import torch_summarize, lr_scheduler
# import pickle

# Learning rate parameters
BASE_LR = 0.01
NUM_CLASSES = 40  # set the number of classes in your dataset
NUM_ATTR = 85
DATA_DIR = "/home/elvis/data/attribute/AwA/Animals_with_Attributes2/zsl/trainval"
BATCH_SIZE = 32
IMAGE_SIZE = 224
# MODEL_NAME = "zsl_resnet18_fc1"
# MODEL_NAME = "zsl_resnet18_fc1_end"
MODEL_NAME = "zsl_resnet18_fc00_awa2"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

parser = argparse.ArgumentParser(description='PyTorch zsl_resnet18_fc00_awa2 Training')
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
    net = attrCNN_awa2(num_attr=NUM_ATTR, num_classes=NUM_CLASSES)

if USE_GPU:
    net.cuda()
    # net = torch.nn.DataParallel(net.module, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

log = open("./log/" + MODEL_NAME + '.txt', 'a')
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


fc_params = list(map(id, net.cnn.fc.parameters()))
base_params = list(filter(lambda p: id(p) not in fc_params, net.cnn.parameters()))

for param in net.parameters():
    param.requires_grad = False
for param in net.cnn.parameters():
    param.requires_grad = True

# optim_params = list(net.cnn.fc.parameters())
# for param in optim_params:
#     param.requires_grad = True
#
# epoch1 = 30
# # optimizer = optim.SGD(optim_params, lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer = optim.Adam(optim_params, weight_decay=0.0005)
# if start_epoch < epoch1:
#     for epoch in range(start_epoch, epoch1):
#         train(epoch, net, optimizer)
#         test(epoch, net)
#     start_epoch = epoch1

# start_epoch = 0
# optimizer = optim.Adam(net.cnn.fc.parameters(), weight_decay=0.0005)
optimizer = optim.Adagrad(net.cnn.parameters(), lr=0.001, weight_decay=0.0005)
# optimizer = torch.optim.SGD([
#     {'params': base_params},
#     {'params': net.cnn.fc.parameters(), 'lr': 1}
# ], lr=1e-4, momentum=0.9, weight_decay=0.0005)

# optimizer = optim.Adam(optim_params, weight_decay=0.0005)
for epoch in range(start_epoch, 500):
    train(epoch, net, optimizer)
    test(epoch, net)
log.close()
