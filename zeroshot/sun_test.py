# -*- coding: utf-8 -*-
"""
@Time    : 2018/1/25 16:06
@Author  : Elvis
"""
"""
CUDA_VISIBLE_DEVICES=0 python cub_test.py
  
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from data.data_loader import DataLoader
from utils.logger import progress_bar


def zsl_test(epoch, net, optimizer, log):
    NUM_CLASSES = 717  # set the number of classes in your dataset
    NUM_SEEN = 645
    NUM_UNSEEN = NUM_CLASSES - NUM_SEEN
    NUM_ATTR = 102
    DATA_DIR = "/home/elvis/data/attribute/SUN/zsl/zsl_test"
    BATCH_SIZE = 32
    IMAGE_SIZE = 224

    best_acc = 67
    USE_GPU = torch.cuda.is_available()
    order_sun_attr = np.load("data/order_sun_attr.npy")
    # w_attr_sum = np.sum(w_attr, 0)
    # w_attr = w_attr/w_attr_sum
    # w_attr[:, 0].sum()
    order_sun_attr = order_sun_attr[NUM_SEEN:, :]
    order_sun_attr = torch.FloatTensor(order_sun_attr).cuda()  # 50 * 312
    net.fc2 = nn.Linear(NUM_ATTR, NUM_CLASSES, bias=False)
    net.fc2.weight = nn.Parameter(order_sun_attr, requires_grad=False)
    # print(torch_summarize(net))
    # print(net)
    net.cuda()
    data_loader = DataLoader(data_dir=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    # train_loader = data_loader.load_data(data_set='train')
    test_loader = data_loader.load_data(data_set='val')
    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    correct_bin = np.zeros(NUM_UNSEEN)
    total_bin = np.zeros(NUM_UNSEEN)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, attr = net(inputs)
        loss = criterion(out, targets)

        test_loss = loss.data[0]
        _, predicted = torch.max(out.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        correct_list = predicted.eq(targets.data).cpu()
        target_list = targets.data.cpu()
        for i, targeti in enumerate(target_list):
            correct_bin[targeti] += correct_list[i]
            total_bin[targeti] += 1.

        acc = 100. * correct / total
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), acc, correct, total))

    acc = 100. * correct / total
    acc_bin = 100. * correct_bin / total_bin
    print("ZSL acc_per_class: %.3f%%(%d/%d)" % (np.mean(acc_bin), correct_bin[0], total_bin[0]))
    log.write(str(np.mean(acc_bin)) + ' ')
    # np.save("data/sun_acc_bin.npy", acc_bin)
    if acc > best_acc:
        MODEL_SAVE_FILE = "zsl_resnet50_sun_epoch%dacc%d.pth" % (epoch, int(np.mean(acc_bin)))
        print(MODEL_SAVE_FILE)
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer
        }
        torch.save(state, "./checkpoints/" + MODEL_SAVE_FILE)


def gzsl_test(epoch, net, optimizer):
    NUM_CLASSES = 717  # set the number of classes in your dataset
    Num_SEEN = 645
    NUM_ATTR = 102
    DATA_DIR = "/home/elvis/data/attribute/SUN/zsl/gzsl_test"
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    best_h = 50
    USE_GPU = torch.cuda.is_available()
    order_sun_attr = np.load("data/order_sun_attr.npy")
    # order_sun_attr[Num_SEEN:, :] = order_sun_attr[Num_SEEN:, :]
    # order_cub_attr = order_cub_attr[150:, :]
    order_sun_attr = torch.FloatTensor(order_sun_attr).cuda()  # 50 * 312
    net.fc2 = nn.Linear(NUM_ATTR, NUM_CLASSES, bias=False)
    net.fc2.weight = nn.Parameter(order_sun_attr, requires_grad=False)
    net.cuda()
    data_loader = DataLoader(data_dir=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    # train_loader = data_loader.load_data(data_set='train')
    test_loader = data_loader.load_data(data_set='val')
    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss, correct_seen, correct_unseen, total_seen, total_unseen, total, loss = 0, 0, 0, 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, attr = net(inputs)
        loss = criterion(out, targets)

        test_loss = loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct_list = predicted.eq(targets.data).cpu()
        target_list = targets.data.cpu()
        for i, targeti in enumerate(target_list):
            if targeti < Num_SEEN:
                correct_seen += correct_list[i]
                total_seen += 1
            else:
                correct_unseen += correct_list[i]
                total_unseen += 1

        acc_seen = 100. * correct_seen / total_seen
        if total_unseen > 0:
            acc_unseen = 100. * correct_unseen / total_unseen
        else:
            acc_unseen = 0.
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | acc_seen: %.3f%% (%d/%d) | acc_unseen: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), acc_seen, correct_seen, total_seen,
                        acc_unseen, correct_unseen, total_unseen))
    acc_seen = 100. * correct_seen / total_seen
    acc_unseen = 100. * correct_unseen / total_unseen
    h = 2./(1./acc_seen + 1./acc_unseen)
    print("acc_seen: %.3f%% (%d/%d) | acc_unseen: %.3f%% (%d/%d) | H: %.3f%%" %
          (acc_seen, correct_seen, total_seen, acc_unseen, correct_unseen, total_unseen, h))
    if h > best_h:
        MODEL_SAVE_FILE = "gzsl_resnet50_sun_epoch%dacc%d.pth" % (epoch, int(h))
        print(MODEL_SAVE_FILE)
        state = {
            'net': net,
            'acc': h,
            'epoch': epoch,
            'optimizer': optimizer
        }
        torch.save(state, "./checkpoints/" + MODEL_SAVE_FILE)


def gzsl_test0(epoch, net, optimizer, log, gamma=2.):
    NUM_CLASSES = 717  # set the number of classes in your dataset
    Num_SEEN = 645
    NUM_ATTR = 102
    DATA_DIR = "/home/elvis/data/attribute/SUN/zsl/gzsl_test"
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    best_h = 55
    USE_GPU = torch.cuda.is_available()
    data_loader = DataLoader(data_dir=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    # train_loader = data_loader.load_data(data_set='train')
    test_loader = data_loader.load_data(data_set='val')
    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss, correct_seen, correct_unseen, total_seen, total_unseen, total, loss = 0, 0, 0, 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, attr = net(inputs)
        loss = criterion(out, targets)

        test_loss = loss.data[0]
        logit = out.data
        # seen_logit = torch.nn.functional.softmax(Variable(logit[:, :150]), dim=1).data
        # unseen_logit = torch.nn.functional.softmax(Variable(logit[:, 150:]), dim=1).data
        seen_prob, seen_class = torch.max(logit[:, :Num_SEEN], 1)
        unseen_prob, unseen_class = torch.max(logit[:, Num_SEEN:], 1)
        predicted = seen_class
        for i, spi in enumerate(seen_prob):
            if seen_prob[i] < unseen_prob[i] * gamma:
                predicted[i] = unseen_class[i] + Num_SEEN
        total += targets.size(0)
        correct_list = predicted.eq(targets.data).cpu()
        target_list = targets.data.cpu()
        for i, targeti in enumerate(target_list):
            if targeti < Num_SEEN:
                correct_seen += correct_list[i]
                total_seen += 1
            else:
                correct_unseen += correct_list[i]
                total_unseen += 1

        acc_seen = 100. * correct_seen / total_seen
        if total_unseen > 0:
            acc_unseen = 100. * correct_unseen / total_unseen
        else:
            acc_unseen = 0.
        progress_bar(batch_idx, len(test_loader),
                     'Loss: %.3f | acc_seen: %.3f%% (%d/%d) | acc_unseen: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), acc_seen, correct_seen, total_seen,
                        acc_unseen, correct_unseen, total_unseen))
    acc_seen = 100. * correct_seen / total_seen
    acc_unseen = 100. * correct_unseen / total_unseen
    h = 2./(1./acc_seen + 1./acc_unseen)
    print("acc_seen: %.3f%% (%d/%d) | acc_unseen: %.3f%% (%d/%d) | H: %.3f%%" %
          (acc_seen, correct_seen, total_seen, acc_unseen, correct_unseen, total_unseen, h))
    log.write(str(acc_seen) + ' ' + str(acc_unseen) + ' ' + str(h) + " ")
    if h > best_h:
        MODEL_SAVE_FILE = "gzsl_resnet50_sun_epoch%dacc%d.pth" % (epoch, int(h))
        print(MODEL_SAVE_FILE)
        state = {
            'net': net,
            'acc': h,
            'epoch': epoch,
            'optimizer': optimizer
        }
        torch.save(state, "./checkpoints/" + MODEL_SAVE_FILE)