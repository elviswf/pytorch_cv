# -*- coding: utf-8 -*-
"""
@Time    : 2018/1/23 17:30
@Author  : Elvis
 recommend.py
  
"""
import os
import numpy as np
import skimage.io
from sklearn import datasets, linear_model
import torch
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
from matplotlib import pyplot as plt


DATA_DIR = "/home/elvis/data/attribute/CUB_200_2011/zsl/trainval"
BASE_LR = 0.01
NUM_CLASSES = 150  # set the number of classes in your dataset
BATCH_SIZE = 32
IMAGE_SIZE = 224
MODEL_NAME = "zsl_resnet101_train"
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
net = checkpoint["net"]
best_acc = checkpoint["acc"]
start_epoch = checkpoint["epoch"]
optimizer = checkpoint["optimizer"]


# weight transfer by
wt = net.fc.weight.data.cpu()
wt_np = wt.cpu().numpy()  # torch.Size([150, 512])
wt_np.shape

cub_attr = np.loadtxt("/home/elvis/data/attribute/CUB_200_2011/zsl/class_attribute_labels_continuous.txt")
cub_attr.shape  # (200, 312)

attr150 = cub_attr[:150, :]
attr150.shape  # (150, 312)

reg = linear_model.Ridge(alpha=0.001)
reg.fit(attr150, wt_np)
w_m = reg.coef_
w_m.shape

attr50 = cub_attr[150:, :]

w_full = w_m.dot(cub_attr.transpose())
wsize = w_full.shape
net.fc = nn.Linear(wsize[0], wsize[1], bias=False)
net.fc.weight = nn.Parameter(torch.FloatTensor(w_full.transpose()))
net.fc.weight.shape

state = {
    'net': net,
    'acc': best_acc,
    'epoch': start_epoch,
    'optimizer': optimizer
}

if not os.path.isdir("checkpoints"):
    os.mkdir('checkpoints')
torch.save(state, "./checkpoints/zsl_resnet101_200.pt")

net = net.cuda()
# net.eval()

centre_crop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


classes = list()
with open("/home/elvis/data/attribute/CUB_200_2011/zsl/classes.txt", "r") as class_file:
    for line in class_file:
        classes.append(line.strip())


img_dir = "/home/elvis/data/attribute/CUB_200_2011/zsl/zsl_test/130.Tree_Sparrow"
for img_name in os.listdir(img_dir)[:6]:
    img_path = os.path.join(img_dir, img_name)
    img = skimage.io.imread(img_path)
    x = Variable(centre_crop(img).unsqueeze(0), volatile=True).cuda()
    logit = net(x)
    h_x = F.softmax(logit).data.cpu().squeeze()
    probs, idx = h_x.sort(0, True)
    print(img_path)
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))


