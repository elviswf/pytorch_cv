# -*- coding: utf-8 -*-
"""
@Time    : 2017/11/5 22:22
@Author  : Elvis
"""
"""
 cifar100.py
  
"""
import torch
import torch.nn.functional as F
from torch import optim
from torch.backends import cudnn

import torchvision
from torchvision import transforms
import os
import argparse
from time import time

from models.resnet8 import ResNet8
from utils.logger import progress_bar
from torch.autograd import Variable
import pickle

# python resnet18.py --mode emb
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--tau', default=2, type=float, help='Softmax temperature')
parser.add_argument('--alpha', default=0.5, type=float, help='alpha')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--mode', default='baseline', type=str, help='baseline or emb or full')
parser.add_argument('--data', default='/home/fwu/code/pytorch/LabelEmb/ComputerVision', type=str,
                    help='file path of the dataset')
parser.add_argument('--num', default='0', type=str)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0.
start_epoch = 0

print("==> Preparing data...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transforms_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.resume:
    print("==> Resuming from checkpoint...")
    assert (os.path.isdir("checkpoint"), "Error, no checkpoint directory found!")
    checkpoint = torch.load("./checkpoint/cifar100_resnet18_" + args.mode + ".pt")
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
else:
    print("==> Building model...")
    net = ResNet8()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

optimizer = optim.Adam(net.parameters())


def my_loss(logit, prob):
    soft_logit = F.log_softmax(logit)
    loss = torch.sum(prob * soft_logit, 1)
    return loss


def comp_loss(out1, out2, tar, emb_w, targets, mode):
    out2_prob = F.softmax(out2)
    tau2_prob = F.softmax(out2 / args.tau).detach()
    soft_tar = F.softmax(tar).detach()
    L_o1_y = F.cross_entropy(out1, targets)
    if mode == "baseline":
        return L_o1_y

    alpha = args.alpha
    _, preds = torch.max(out2, 1)
    mask = preds.eq(targets).float().detach()
    L_o1_emb = - torch.mean(my_loss(out1, soft_tar))
    if mode == 'emb':
        return alpha * L_o1_y + (1 - alpha) * L_o1_emb

    L_o2_y = F.cross_entropy(out2, targets)
    L_emb_o2 = -torch.sum(my_loss(tar, tau2_prob) * mask) / (torch.sum(mask) + 1e-8)
    gap = torch.gather(out2_prob, 1, targets.view(-1, 1)) - 0.9
    L_re = torch.sum(F.relu(gap))
    # L2_loss = F.mse_loss(emb_w.t(), emb_w.detach())

    loss = alpha * L_o1_y + (1 - alpha) * L_o1_emb + L_o2_y + L_emb_o2 + L_re
    return loss


print(args.mode)
if not os.path.isdir("10_results"):
    os.mkdir("10_results")

log = open("./10_results/" + args.num + args.mode + '.txt', 'a')


def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets, requires_grad=False)
        out1, out2, tar, emb_w = net(inputs, targets, epoch, batch_idx)

        loss = comp_loss(out1, out2, tar, emb_w, targets, args.mode)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(out1.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(train_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d"
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    log.write(str(epoch) + ' ' + str(correct / total) + ' ')
    if args.mode != "baseline":
        pickle.dump(emb_w.data.cpu().numpy(), open("./10_results/" + args.num + args.mode + "embedding.pkl", "wb"))


def test(epoch):
    global best_acc
    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, _, _, _ = net(inputs, targets, -1, batch_idx)
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
    if epoch > 30 and acc > best_acc:
        print("Saving checkpoint")
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir('checkpoint')
        torch.save(state, "./checkpoint/cifar100_resnet18_" + args.mode + ".pt")
        best_acc = acc


for epoch in range(start_epoch, 100):
    train(epoch)
    test(epoch)
