# -*- coding: utf-8 -*-
"""
@Time    : 2017/11/20 15:04
@Author  : Elvis
"""
"""
 cub.py
  
"""
import torch
import torch.nn.functional as F
from torch import optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision
import os
import argparse
from globalConfig import *
from data.data_loader import DataLoader
from models.emb_resnet import EmbResNet
from utils.logger import progress_bar
from utils.param_count import torch_summarize, lr_scheduler
import pickle

# python cub.py --data /home/fwu/code/pytorch/pytorch-mini/datasets/ --epochs 64
parser = argparse.ArgumentParser(description='PyTorch cub_emb Training')
parser.add_argument('--lr', default=BASE_LR, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--tau', default=2, type=float, help='Softmax temperature')
parser.add_argument('--alpha', default=0.5, type=float, help='alpha')
parser.add_argument('--mode', default='emb', type=str, help='baseline or emb or full')
parser.add_argument('--data', default=DATA_DIR, type=str, help='file path of the dataset')
parser.add_argument('--epochs', default=NUM_EPOCHS, type=int, help='training epochs')
args = parser.parse_args()

best_acc = 0.
start_epoch = 0

if args.resume:
    print("==> Resuming from checkpoint...")
    checkpoint = torch.load("./checkpoints/" + args.mode + "_" + MODEL_SAVE_FILE)
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    optimizer = checkpoint["optimizer"]
else:
    print("==> Building model...")
    net = EmbResNet(num_classes=200)
    optimizer = optim.Adam(net.parameters())


optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# print(torch_summarize(net))
# print(net)
print("mode: " + args.mode)
if USE_GPU:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


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


log = open("./log/" + args.mode + "_" + MODEL_NAME + '_cub.txt', 'a')
print("==> Preparing data...")
data_loader = DataLoader(data_dir=args.data, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
inputs, classes = next(iter(data_loader.load_data()))
out = torchvision.utils.make_grid(inputs)
data_loader.show_image(out, title=[data_loader.data_classes[c] for c in classes])
train_loader = data_loader.load_data(data_set='train')
test_loader = data_loader.load_data(data_set='val')


def train(epoch, optimizer):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, init_lr=0.0008, decay_epoch=start_epoch)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if USE_GPU:
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

        progress_bar(batch_idx, len(train_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    log.write(str(epoch) + ' ' + str(correct / total) + ' ')
    if args.mode != "baseline":
        pickle.dump(emb_w.data.cpu().numpy(),
                    open("./log/embed/" + MODEL_NAME + '_' + args.mode + "embedding.pkl", "wb"))


def test(epoch):
    global best_acc
    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if USE_GPU:
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
    if epoch > 15 and acc > best_acc:
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
