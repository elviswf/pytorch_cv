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
from models.spatial_transform import stnResnet
from utils.logger import progress_bar
import pickle

# python cub.py --data /home/fwu/code/pytorch/pytorch-mini/datasets/ --epochs 64
parser = argparse.ArgumentParser(description='PyTorch cub Training')
parser.add_argument('--lr', default=BASE_LR, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--data', default=DATA_DIR, type=str, help='file path of the dataset')
parser.add_argument('--epochs', default=NUM_EPOCHS, type=int, help='training epochs')
args = parser.parse_args()

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
    net = stnResnet(num_classes=NUM_CLASSES)
    # net = STN2Resnet(num_classes=NUM_CLASSES)

# print(net)
if USE_GPU:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

# optimizer = optim.Adam(net.parameters())
ml = list()
ml.append({'params': net.localization.parameters(), 'lr': 1e-4*BASE_LR})
ml.append({'params': net.fc_loc.parameters(), 'lr': 1e-4*BASE_LR})
ml.append({'params': net.cnn1.parameters(), 'lr': BASE_LR})
ml.append({'params': net.cnn2.parameters(), 'lr': BASE_LR})
ml.append({'params': net.linear.parameters(), 'lr': BASE_LR})
optimizer = optim.SGD(ml, lr=BASE_LR, weight_decay=1e-5)
# print("# trainable parameters: %d" % net.parameters())


def my_loss(logit, prob):
    soft_logit = F.log_softmax(logit)
    loss = torch.sum(prob * soft_logit, 1)
    return loss


def comp_loss(out, targets, mode="baseline"):
    # out2_prob = F.softmax(out2)
    # tau2_prob = F.softmax(out2 / args.tau).detach()
    # tar = emb(targets)
    # soft_tar = F.softmax(tar).detach()
    L_o_y = F.cross_entropy(out, targets)
    return L_o_y

    # alpha = 0.5
    # _, preds = torch.max(out, 1)
    # mask = preds.eq(targets).float().detach()
    # L_o1_emb = - torch.mean(my_loss(out, soft_tar))
    # if mode == 'emb':
    #     return alpha * L_o1_y + (1 - alpha) * L_o1_emb


log = open("./log/" + MODEL_NAME + str(args.epochs) + '_cub.txt', 'a')
print("==> Preparing data...")
data_loader = DataLoader(data_dir=args.data, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
inputs, classes = next(iter(data_loader.load_data()))
out = torchvision.utils.make_grid(inputs)
data_loader.show_image(out, title=[data_loader.data_classes[c] for c in classes])
train_loader = data_loader.load_data(data_set='train')
test_loader = data_loader.load_data(data_set='val')


def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets, requires_grad=False)
        out, theta, _ = net(inputs)

        if batch_idx % 60 == 0:
            print(theta.cpu()[0])
        mode = "baseline"
        loss = comp_loss(out, targets, mode)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(train_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d"
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    log.write(str(epoch) + ' ' + str(correct / total) + ' ')
    # if args.mode != "baseline":
    #     pickle.dump(emb_w.data.cpu().numpy(), open("./log/" + MODEL_NAME + '_' + args.mode + "embedding.pkl", "wb"))


def test(epoch):
    global best_acc
    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, theta, x1 = net(inputs)
        # img = data_loader.un_normalize(x1.data[0].cpu())
        # to_pil = torchvision.transforms.ToPILImage()
        # img = to_pil(img)
        # img.save("imgs/grid/batch_%d.jpg" % batch_idx)
        loss = F.cross_entropy(out, targets)

        test_loss = loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        acc = 100. * correct / total
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), acc, correct, total))
    log.write(str(correct / total) + ' ' + str(test_loss) + '\n')
    log.write(str(theta.cpu()[0]) + '\n')
    log.flush()
    img = data_loader.un_normalize(x1.data.cpu()[0])
    to_pil = torchvision.transforms.ToPILImage()
    img = to_pil(img)
    img.save("imgs/grid/test_%d.jpg" % epoch)

    acc = 100. * correct / total
    if epoch > 30 and acc > best_acc:
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


for epoch in range(start_epoch, 200):
    train(epoch)
    test(epoch)
