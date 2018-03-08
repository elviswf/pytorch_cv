# -*- coding: utf-8 -*-
"""
@Time    : 2018/2/5 14:28
@Author  : Elvis
 modelprint.py
  
"""
import torch

MODEL_NAME = "gzsl_resnet50_gs2"
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
fc0_w = fc1[0].weight.data
fc0_b = fc1[0].bias.data

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
np.save("x_attr_gs2.npy", x_attr)

x_attr

h_x = F.softmax(logit).data.squeeze()
probs, idx = h_x.sort(0, True)

import torch
from torch import nn
from torchvision.models import resnet50


class ResNet50(nn.Module):
    def __init__(self, ):
        super(ResNet50, self).__init__()

        model = resnet50(pretrained=True)
        size = model.fc.weight.size()
        model.fc = nn.Linear(size[1], size[0], bias=False)
        self.model = model

    def forward(self, x):
        return self.model(x)


MODEL_SAVE_FILE = "model_best.pth.tar"
print("==> Resuming from checkpoint...")
checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
net = torch.nn.DataParallel(ResNet50()).cuda()
net.load_state_dict(checkpoint['state_dict'])

w = net.module.model.fc.weight
w_np = w.data.cpu().numpy()
w_np.shape
np.save("data/w_np.npy", w_np)

import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

w_np = np.load("data/w_np.npy")
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})
RS = 20180305


def tsne_w(Xw):
    model = TSNE(n_components=2, random_state=RS)
    np.set_printoptions(suppress=True)
    emb = model.fit_transform(Xw)
    return emb


# Scale and visualize the embedding vectors
def scatter(x, colors, txts=None):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", colors.size))

    # We create a scatter plot.
    plt.figure(figsize=(16, 16))
    ax = plt.subplot(aspect='equal')
    ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.title("1000*2048 -> 1000*2 embedding")

    for i in range(colors.size):
        # Position of each label.
        # if (i % 8 == 0):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        if len(txts[i]) > 10:
            txts[i] = txts[i][:8]
        txt = ax.text(xtext, ytext, txts[i], fontsize=6)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=3, foreground="w"),
        PathEffects.Normal()])

    plt.savefig("data/embedding/weight/resnet50_fc_no_bias.pdf")
    return


nclass = 1000
yc = np.array(list(range(nclass)))
import os
file_name = 'data/synset_words.txt'
if not os.access(file_name, os.W_OK):
    synset_URL = 'https://github.com/szagoruyko/functional-zoo/raw/master/synset_words.txt'
    os.system('wget ' + synset_URL + ' -P data/')

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ', 1)[1].split(', ', 1)[0])

emb_w = tsne_w(w_np)
scatter(emb_w, colors=yc, txts=classes)
