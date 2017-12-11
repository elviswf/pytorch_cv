# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/10 16:04
@Author  : Elvis
"""
"""
 cifar10_base.py
  
"""
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn

from torchvision import datasets, transforms
import os
import argparse

from models.densenet_efficient import DenseNetEfficient
from utils.logger import progress_bar
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=96, metavar='N',
                    help='input batch size for testing (default: 512)')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--data', default='/home/fwu/code/pytorch/data/cifar10/', type=str,
                    help='file path of the dataset')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

NUM_CLASSES = 10
MODEL_SAVE_FILE = "cifar10_densenet.pt"
best_acc = 0.
start_epoch = 0

if args.resume:
    print("==> Resuming from checkpoint...")
    checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
else:
    print("==> Building model...")
    net = DenseNetEfficient()

log = open("./log/cifar10_densenet.txt", 'a')

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

train_set = datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

test_set = datasets.CIFAR10(root=args.data, train=False, download=True, transform=transforms_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.cuda:
    net.cuda()

optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()


def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (batch_x, target) in enumerate(train_loader):
        if args.cuda:
            batch_x, target = batch_x.cuda(), target.cuda()
        batch_x, target = Variable(batch_x), Variable(target, requires_grad=False)
        optimizer.zero_grad()
        output = net(batch_x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, pred = torch.max(output.data, 1)
        total += target.size(0)
        correct += pred.eq(target.data).cpu().sum()
        progress_bar(batch_idx, len(train_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (batch_x, target) in enumerate(test_loader):
        if args.cuda:
            batch_x, target = batch_x.cuda(), target.cuda()
        batch_x, target = Variable(batch_x, volatile=True), Variable(target, volatile=True, requires_grad=False)
        output = net(batch_x)
        loss = criterion(output, target)
        test_loss += loss.data[0]
        _, pred = torch.max(output.data, 1)
        total += target.size(0)
        correct += pred.eq(target.data).cpu().sum()
        progress_bar(batch_idx, len(test_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    log.write("Test Epoch %d: acc: %.3f | loss: %.3f\n" % (epoch, correct / total, test_loss))
    log.flush()

    acc = 100. * correct / len(test_loader.dataset)
    if (epoch > 5 or acc > 50.) and acc > best_acc:
        print("Saving checkpoint")
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir("checkpoints"):
            os.mkdir('checkpoints')
        torch.save(state, "./checkpoints/" + MODEL_SAVE_FILE)
        best_acc = acc


for epoch in range(start_epoch + 1, args.epochs + 1):
    train(epoch)
    test(epoch)

log.close()