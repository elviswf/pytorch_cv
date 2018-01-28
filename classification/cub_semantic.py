# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/4 15:42
@Author  : Elvis
"""
"""
CUDA_VISIBLE_DEVICES=4 python cub_fusion.py
  
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision
import os
import argparse
from data.data_loader import DataLoader
from models.semantic import semanticResnet, MSELoss, predict_y
from utils.logger import progress_bar
from utils.param_count import torch_summarize, lr_scheduler
import pickle

# Learning rate parameters
BASE_LR = 0.01
# DATASET INFO
NUM_CLASSES = 200  # set the number of classes in your dataset
DATA_DIR = "/home/elvis/code/data/cub200"
BATCH_SIZE = 32
IMAGE_SIZE = 224
MODEL_NAME = "semanticResnet"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

parser = argparse.ArgumentParser(description='PyTorch vgg16_sp Training')
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
    net = semanticResnet(NUM_CLASSES)

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
out = torchvision.utils.make_grid(inputs)
data_loader.show_image(out, title=[data_loader.data_classes[c] for c in classes])
train_loader = data_loader.load_data(data_set='train')
test_loader = data_loader.load_data(data_set='val')
criterion = MSELoss()


def y_encoding(y):
    y = y.view((-1, 1))
    one_hot = torch.FloatTensor(y.size(0), NUM_CLASSES).zero_()
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
        y_emb = y_encoding(targets)
        if USE_GPU:
            inputs, y_emb = inputs.cuda(), y_emb.cuda()
        inputs, y_emb = Variable(inputs, requires_grad=True), Variable(y_emb, requires_grad=True)
        optimizer.zero_grad()

        v_x, v_y = net(inputs, y_emb)
        loss = criterion(v_x, v_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        # y_pred = predict_y(v_x, v_y)
        total += targets.size(0)
        # correct += y_pred.squeeze().cpu().eq(targets.data).sum()

        progress_bar(batch_idx, len(train_loader), "Loss: %.3f "% (train_loss / (batch_idx + 1)))

    log.write(str(epoch) + ' ' + str(correct / total) + ' ')


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    y_all = torch.arange(0, NUM_CLASSES, out=torch.LongTensor())
    y_code = y_encoding(y_all)
    test_x = torch.zeros(200, 3, 224, 224)
    if USE_GPU:
        test_x, y_code = test_x.cuda(), y_code.cuda()
    test_x, y_code = Variable(test_x, volatile=True), Variable(y_code, volatile=True)
    _, v_y_all = net(test_x, y_code)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        y_emb = y_encoding(targets)
        if USE_GPU:
            inputs, y_emb = inputs.cuda(), y_emb.cuda()
        inputs, y_emb = Variable(inputs, volatile=True), Variable(y_emb, volatile=True)
        v_x, v_y = net(inputs, y_emb)
        loss = criterion(v_x, v_y)
        test_loss = loss.data[0]

        y_pred = predict_y(v_x, v_y_all)
        total += targets.size(0)
        correct += y_pred.data.cpu().eq(targets).sum()

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

optim_params = list(net.y2v.parameters())
for param in optim_params:
    param.requires_grad = True

epoch1 = 100
if start_epoch < epoch1:
    optimizer = optim.SGD(optim_params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    for epoch in range(start_epoch, epoch1):
        train(epoch, net, optimizer)
        test(epoch, net)
    start_epoch = epoch1

# optim_params = list(net.parameters())
# for param in optim_params:
#     param.requires_grad = True
#
# optimizer = optim.SGD(optim_params, lr=0.001, momentum=0.9, weight_decay=0.0005)
# for epoch in range(start_epoch, 100):
#     train(epoch, net, optimizer)
#     test(epoch, net)
log.close()
