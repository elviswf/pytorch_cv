# -*- coding: utf-8 -*-
"""
@Time    : 2018/2/5 14:28
@Author  : Elvis
 modelprint.py
  
"""
import torch
from torchvision.models import alexnet
MODEL_NAME = "gzsl_resnet50_fc051"
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
# print("optimizer: %f" % optimizer)

fc0 = net.fc0
fc1 = net.fc1
fc0_w = fc0[0].weight.data
fc0_b = fc0[0].bias.data

import matplotlib.pyplot as plt
# matplotlib.use("Agg")
import skimage.io
import os

file_name = '/home/elvis/code/data/cub200/val/075.Green_Jay/Green_Jay_0130_65885.jpg'

img = skimage.io.imread(file_name)
plt.ion()
plt.imshow(img)
plt.savefig()

from torchvision import transforms

centre_crop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from torch.autograd import Variable
from torch.nn import functional as F

# get top 5 probabilities
x = Variable(centre_crop(img).unsqueeze(0), volatile=True).cuda()
logit, attr = net(x)

x_attr = attr.cpu().data.numpy()

import numpy as np
np.save("x_attr.npy", x_attr)

h_x = F.softmax(logit).data.squeeze()
probs, idx = h_x.sort(0, True)
