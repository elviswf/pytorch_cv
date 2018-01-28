# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/4 15:42
@Author  : Elvis

 cub.py
watch --color -n1 gpustat -cpu
CUDA_VISIBLE_DEVICES=3 python cub_attr1.py

zsl_resnet18_fc00 : Sigmoid + dropout 0.5  74.789% (1329/1777)  ZSL_Acc: 53.354% (1583/2967)  200 epoch
zsl_resnet18_fc01 : Sigmoid with fc pretrain   Acc: 73.044% (1298/1777)  ZSL_Acc: 24.537% (728/2967)
zsl_resnet18_fc02 : Sigmoid with fc pretrain + dropout 0.5 full 150   60 epoch:  Acc: 50.792% (1507/2967)
zsl_resnet18_fc03 : Sigmoid + dropout 0.5 weight_decay=0.005 full 150   60 epoch:  Acc: 50.792% (1507/2967)
                     100 epoch:    Acc: 53.758% (1595/2967)    100 epoch: Acc: 54.297% (1611/2967)


"""
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.autograd import Variable
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
DATA_DIR = "/home/elvis/data/attribute/CUB_200_2011/zsl/trainval0"
BATCH_SIZE = 32
IMAGE_SIZE = 224
# MODEL_NAME = "zsl_resnet18_fc1"
# MODEL_NAME = "zsl_resnet18_fc1_end"
MODEL_NAME = "zsl_resnet18_fc03"
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
if args.resume:
    print("==> Resuming from checkpoint...")
    checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    optimizer = checkpoint["optimizer"]
else:
    print("==> Building model...")
    net = attrCNN(num_attr=312, num_classes=150)


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
criterion = nn.CrossEntropyLoss()


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

# optim_params = list(net.cnn.fc.parameters())
# for param in optim_params:
#     param.requires_grad = True

# epoch1 = 20
# optimizer = optim.Adagrad(optim_params, lr=0.01, weight_decay=0.0005)
# optimizer = optim.Adam(optim_params, weight_decay=0.0005)
# if start_epoch < epoch1:
#     for epoch in range(start_epoch, epoch1):
#         train(epoch, net, optimizer)
#         test(epoch, net)
#     start_epoch = epoch1
for param in net.cnn.parameters():
    param.requires_grad = True

# optimizer = optim.Adagrad(net.cnn.parameters(), lr=0.001, weight_decay=0.005)
# start_epoch = 0
# optimizer = optim.Adam(net.cnn.fc.parameters(), weight_decay=0.0005)
# optimizer = torch.optim.SGD([
#     {'params': base_params},
#     {'params': net.cnn.fc.parameters(), 'lr': 1}
# ], lr=1e-4, momentum=0.9, weight_decay=0.0005)
from zeroshot.cub_test import zsl_test
import copy
# optimizer = optim.Adam(optim_params, weight_decay=0.0005)
for epoch in range(start_epoch, 500):
    train(epoch, net, optimizer)
    test(epoch, net)
    net1 = copy.deepcopy(net)
    zsl_test(epoch, net1, optimizer)
log.close()
