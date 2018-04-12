# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/4 15:42
@Author  : Elvis

 cub.py
watch --color -n1 gpustat -cpu
CUDA_VISIBLE_DEVICES=2 python sun

zsl_resnet18_fc00 : Sigmoid + dropout 0.5  74.789% (1329/1777)  ZSL_Acc: 53.354% (1583/2967)  200 epoch
zsl_resnet18_fc01 : Sigmoid with fc pretrain   Acc: 73.044% (1298/1777)  ZSL_Acc: 24.537% (728/2967)
zsl_resnet18_fc02 : Sigmoid with fc pretrain + dropout 0.5 full 150   60 epoch:  Acc: 50.792% (1507/2967)
zsl_resnet18_fc03 : Sigmoid + dropout 0.5 weight_decay=0.005 full 150   60 epoch:  Acc: 50.792% (1507/2967)
                     100 epoch:    Acc: 53.758% (1595/2967)    192 epoch: Acc: 54.803% (1626/2967)

zsl_resnet50_fc00 : Sigmoid + dropout 0.5 weight_decay=0.005 full 150 44epoch Acc: 57.162% (1696/2967)
                                            Acc: 75.842% (6690/8821) | Test Acc: 95.948% (1705/1777)
gzsl_resnet50_fc01 : Sigmoid + dropout 0.5
 Step: 246ms | Tot: 40s814ms | Loss: 0.068 | Acc: 98.764% (8712/8821)
 Step: 73ms | Tot: 3s593ms | Loss: 0.000 | Acc: 100.000% (1777/1777)
 Step: 36ms | Tot: 6s308ms | Loss: 0.026 | Acc: 66.903% (1985/2967)  Epoch: 26

gzsl_resnet50_fc03 : fc2 drop sigmoid     Acc: 67.914% (2015/2967)   gzsl:  Acc: 76.044% (4406/5794)
gzsl_resnet50_fc04 :  fc2 dropout no good   Acc: 64.948% (1927/2967)   Acc: 65.285% (1937/2967)
gzsl_resnet50_fc05:  fc2 drop sigmoid      gzsl  Acc: 32.652% (1549/4744)  no sigmoid Acc: 61.139% (1814/2967)
gzsl_resnet50_fc06: squeeze net
"""
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.autograd import Variable
import os
import argparse
from data.data_loader import DataLoader
from models.zsl_resnet import attrWCNNg_sun, RegLoss
# from models.focalLoss import FocalLoss
from zeroshot.sun_test import zsl_test, gzsl_test0, gzsl_test
from utils.logger import progress_bar
# from utils.param_count import torch_summarize, lr_scheduler
# import pickle

# Learning rate parameters
BASE_LR = 0.01
NUM_CLASSES = 717  # set the number of classes in your dataset
NUM_ATTR = 102
DATA_DIR = "/home/elvis/data/attribute/SUN/zsl/trainval"
BATCH_SIZE = 32
IMAGE_SIZE = 224
# MODEL_NAME = "zsl_resnet18_fc1"
# MODEL_NAME = "zsl_resnet18_fc1_end"
gamma = 1.4
lamda2 = 0.1
MODEL_NAME = "sun_gzsl_g1_g14d"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

parser = argparse.ArgumentParser(description='PyTorch sun_gzsl_resnet50_g_g17 Training')
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
    net = attrWCNNg_sun(num_attr=NUM_ATTR, num_classes=NUM_CLASSES)

# print(torch_summarize(net))
# print(net)
if USE_GPU:
    net.cuda()
    # net = torch.nn.DataParallel(net.module, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

log = open("./log/" + MODEL_NAME + '_sun.txt', 'a')
print("==> Preparing data...")
data_loader = DataLoader(data_dir=args.data, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
inputs, classes = next(iter(data_loader.load_data()))
# out = torchvision.utils.make_grid(inputs)
# data_loader.show_image(out, title=[data_loader.data_classes[c] for c in classes])
train_loader = data_loader.load_data(data_set='train')
test_loader = data_loader.load_data(data_set='val')
# criterion = nn.CrossEntropyLoss()
criterion = RegLoss(lamda2=lamda2, superclass="sun")
# criterion = FocalLoss(class_num=NUM_CLASSES, gamma=0)


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


epoch1 = 3
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
    start_epoch = epoch1

fc_params = list(map(id, net.fc2.parameters()))
base_params = list(filter(lambda p: id(p) not in fc_params, net.parameters()))
for param in base_params:
    param.requires_grad = True

optimizer = optim.Adagrad(base_params, lr=0.001, weight_decay=0.0005)
import copy
for epoch in range(start_epoch, 100):
    train(epoch, net, optimizer)
    test(epoch, net)
    gzsl_test0(epoch, net, optimizer, log, gamma=gamma)
    net1 = copy.deepcopy(net)
    zsl_test(epoch, net1, optimizer, log)
    del net1

log.close()
