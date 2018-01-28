# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/4 15:42
@Author  : Elvis
"""
"""
 cub.py
  
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
from models.bilinear import cnnbilinear
from torchvision.models import resnet18
from utils.logger import progress_bar
from utils.param_count import torch_summarize, lr_scheduler
import pickle

# Learning rate parameters
BASE_LR = 0.001

# DATASET INFO
NUM_CLASSES = 200 # set the number of classes in your dataset
DATA_DIR = "/home/fwu/code/pytorch/data/cub200"
BATCH_SIZE = 32
IMAGE_SIZE = 224
MODEL_NAME = "cub_bilinear_f1"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'
best_acc = 0.
start_epoch = 0
num_classes = 200
NUM_EPOCHS = 200

# python cub.py --data /home/fwu/code/pytorch/pytorch-mini/datasets/ --epochs 64
parser = argparse.ArgumentParser(description='PyTorch cub_emb Training')
parser.add_argument('--lr', default=BASE_LR, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--data', default=DATA_DIR, type=str, help='file path of the dataset')
parser.add_argument('--epochs', default=NUM_EPOCHS, type=int, help='training epochs')
args = parser.parse_args()


if args.resume:
    print("==> Resuming from checkpoint...")
    checkpoint = torch.load("./checkpoints/" + "_" + MODEL_SAVE_FILE)
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    optimizer = checkpoint["optimizer"]
else:
    print("==> Building model...")
    net = cnnbilinear()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

# optimizer = optim.Adadelta(net.parameters(), lr=0.01, weight_decay=0.0005)
# for param_group in optimizer.param_groups:
#         param_group['lr'] = 0.002
# optimizer = optim.Adam(net.parameters(), weight_decay=0.0005)
# print(torch_summarize(net))
# print(net)
if torch.cuda.is_available():
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


log = open("./log/" + "_" + MODEL_NAME + '_cub.txt', 'a')
print("==> Preparing data...")
data_loader = DataLoader(data_dir=args.data, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
inputs, classes = next(iter(data_loader.load_data()))
out = torchvision.utils.make_grid(inputs)
data_loader.show_image(out, title=[data_loader.data_classes[c] for c in classes])
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
        loss = F.cross_entropy(out, targets)

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
        torch.save(state, "./checkpoints/" + "_" + MODEL_SAVE_FILE)
        best_acc = acc

#
for param in net.parameters():
    param.requires_grad = False
optim_params = list(net.cnn2.parameters()) + list(net.classifier.parameters())
for param in optim_params:
    param.requires_grad = True

# optimizer = optim.Adam(optim_params, weight_decay=0.0005)
# for epoch in range(start_epoch, 40):
#     train(epoch, net, optimizer)
#     test(epoch, net)
# for param in net.cnn1.parameters():
#     param.requires_grad = False
#
optimizer = optim.Adadelta(optim_params, weight_decay=0.0005)
for epoch in range(start_epoch, 200):
    train(epoch, net, optimizer)
    test(epoch, net)

log.close()
