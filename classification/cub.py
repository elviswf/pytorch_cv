# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/4 15:42
@Author  : Elvis
"""
"""
 cub.py
watch --color -n1 gpustat -cpu
"""
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision
import os
import argparse
from globalConfig import *
from data.data_loader import DataLoader
from torchvision.models import resnet18
from utils.logger import progress_bar
from utils.param_count import torch_summarize, lr_scheduler
import pickle

# python cub.py --data /home/fwu/code/pytorch/pytorch-mini/datasets/ --epochs 64
parser = argparse.ArgumentParser(description='PyTorch cub_emb Training')
parser.add_argument('--lr', default=BASE_LR, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--tau', default=2, type=float, help='Softmax temperature')
parser.add_argument('--alpha', default=0.5, type=float, help='alpha')
parser.add_argument('--mode', default='baseline', type=str, help='baseline or emb or full')
parser.add_argument('--data', default=DATA_DIR, type=str, help='file path of the dataset')
parser.add_argument('--epochs', default=NUM_EPOCHS, type=int, help='training epochs')
args = parser.parse_args()

best_acc = 0.
start_epoch = 0
num_classes = 200

if args.resume:
    print("==> Resuming from checkpoint...")
    checkpoint = torch.load("./checkpoints/" + args.mode + "_" + MODEL_SAVE_FILE)
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    optimizer = checkpoint["optimizer"]
else:
    print("==> Building model...")
    net = resnet18(pretrained=True)
    net.fc = nn.Linear(512, num_classes)
    # optimizer = optim.Adam(net.parameters())

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
# print(torch_summarize(net))
# print(net)
print("mode: " + args.mode)
if USE_GPU:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


log = open("./log/" + args.mode + "_" + MODEL_NAME + '_cub.txt', 'a')
print("==> Preparing data...")
data_loader = DataLoader(data_dir=args.data, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
inputs, classes = next(iter(data_loader.load_data()))
out = torchvision.utils.make_grid(inputs)
data_loader.show_image(out, title=[data_loader.data_classes[c] for c in classes])
train_loader = data_loader.load_data(data_set='train')
test_loader = data_loader.load_data(data_set='val')
criterion = nn.CrossEntropyLoss()


def train(epoch, optimizer):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, init_lr=0.001, decay_epoch=start_epoch)
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


def test(epoch):
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
        torch.save(state, "./checkpoints/" + args.mode + "_" + MODEL_SAVE_FILE)
        best_acc = acc


for epoch in range(start_epoch, 400):
    train(epoch, optimizer)
    test(epoch)

log.close()
