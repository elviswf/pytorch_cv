# -*- coding: utf-8 -*-
"""
@Time    : 2018/2/5 14:28
@Author  : Elvis
 modelprint.py
  
"""
import torch

MODEL_NAME = "binearAR_resnet50"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

print("Model: " + MODEL_NAME)
print("==> Resuming from checkpoint...")
checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
net = checkpoint["net"]
best_acc = checkpoint["acc"]
start_epoch = checkpoint["epoch"]
optimizer = checkpoint["optimizer"]


print(net)
print("best_acc: %f" % best_acc)
print("start_epoch: %f" % start_epoch)
print("optimizer: %f" % optimizer)