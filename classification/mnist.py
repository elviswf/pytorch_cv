# -*- coding: utf-8 -*-
"""
@Time    : 2017/11/26 21:50
@Author  : Elvis
"""
"""
 mnist.py
  
"""
import argparse
import torch
import os
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.capsule import CapsuleNet, CapsuleLoss, augmentation
from utils.logger import progress_bar

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=96, metavar='N',
                    help='input batch size for testing (default: 512)')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
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

NUM_CLASSES = 10
MODEL_SAVE_FILE = "mnist_capsule.pt"
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
    net = CapsuleNet()

log = open("./log/mnist_capsule.txt", 'a')
print("==> Preparing data...")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

if args.cuda:
    net.cuda()

optimizer = optim.Adam(net.parameters())
capsuleLoss = CapsuleLoss()


def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (batch_x, target) in enumerate(train_loader):
        batch_x = augmentation(batch_x)
        target = torch.LongTensor(target)
        labels = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
        if args.cuda:
            batch_x, labels = batch_x.cuda(), labels.cuda()
        batch_x, labels = Variable(batch_x), Variable(labels)
        optimizer.zero_grad()
        classes, reconstructions = net(batch_x, labels)
        loss = capsuleLoss(batch_x, labels, classes, reconstructions)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        total += target.size(0)
        pred = classes.data.max(1, keepdim=True)[1].cpu()
        correct += pred.eq(target.view_as(pred)).sum()
        progress_bar(batch_idx, len(train_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (batch_x, target) in enumerate(test_loader):
        target = torch.LongTensor(target)
        labels = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
        if args.cuda:
            batch_x, labels = batch_x.cuda(), labels.cuda()
        batch_x, labels = Variable(batch_x, volatile=True), Variable(labels)
        classes, reconstructions = net(batch_x)
        loss = capsuleLoss(batch_x, labels, classes, reconstructions)
        test_loss += loss.data[0]
        total += target.size(0)
        pred = classes.data.max(1, keepdim=True)[1].cpu()
        correct += pred.eq(target.view_as(pred)).sum()
        progress_bar(batch_idx, len(test_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    log.write("Test Epoch %d: acc: %.3f | loss: %.3f\n" % (epoch, correct / total, test_loss))
    log.flush()

    acc = 100. * correct / len(test_loader.dataset)
    if epoch > 0 and acc > best_acc:
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