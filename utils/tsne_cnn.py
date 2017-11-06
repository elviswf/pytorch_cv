# -*- coding: utf-8 -*-
"""
@Time    : 2017/2/24 10:13
@Author  : Elvis

weight
(1000, 4096)
(1000, 4096)
(1000, 4096)
(1000, 2048)
(1000, 512)
(1000, 512)
(1000, 2048)
(1000, 2048)
(1000, 2048)
"""
import numpy as np
from sklearn.manifold import TSNE
from torchvision import models
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['axes.unicode_minus'] = False
RS = 20171106


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def get_wb(name="alexnet"):
    model = models.__dict__[name](pretrained=True)
    if name == "alexnet" or "vgg" in name:
        weight = model.classifier[-1].weight.data.numpy()
        bias = model.classifier[-1].bias.data.numpy()
    else:
        weight = model.fc.weight.data.numpy()
        bias = model.fc.bias.data.numpy()
    return weight, bias


models_list = ['alexnet', 'vgg16', 'vgg19', 'inception_v3', 'resnet18', 'resnet34', 'resnet50',
               'resnet101', 'resnet152']
"""
weight_dict = dict()
bias_dict = dict()
for name in models_list:
    weight, bias = get_wb(name)
    print(weight.shape)
    weight_dict[name] = weight
    bias_dict[name] = bias

with open('data/weight_dict.pkl', 'wb') as fw:
    pickle.dump(weight_dict, fw)

with open('data/bias_dict.pkl', 'wb') as fw:
    pickle.dump(bias_dict, fw)
"""
with open('data/weight_dict.pkl', 'rb') as fr:
    weight_dict = pickle.load(fr)
with open('data/bias_dict.pkl', 'rb') as fr:
    bias_dict = pickle.load(fr)

import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})


def tsne_w(weights, name):
    Xw = weights[name]
    model = TSNE(n_components=2, random_state=RS)
    np.set_printoptions(suppress=True)
    emb = model.fit_transform(Xw)
    return emb


# Scale and visualize the embedding vectors
def scatter(x, colors, txts=None, name="alexnet"):
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
    plt.title("%s_w %s embedding" % (name, str(weight_dict[name].shape)))

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

    plt.savefig("data/embedding/weight/%s_w.png" % name)
    return


# img = Image.fromarray(Xw.transpose(), 'RGB')
# img.save('w8n.png')
# img.show()
nclass = 1000
cntxts = open("data/synset_words_cn.txt").readlines()
cntxts = [line.strip() for line in cntxts]

yc = np.array(list(range(nclass)))

# get classes
import os

file_name = 'data/synset_words.txt'
if not os.access(file_name, os.W_OK):
    synset_URL = 'https://github.com/szagoruyko/functional-zoo/raw/master/synset_words.txt'
    os.system('wget ' + synset_URL + ' -P data/')

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ', 1)[1].split(', ', 1)[0])

# emb_w = dict()
# for name in models_list:
#     emb_w[name] = tsne_w(weight_dict, name)
#     scatter(emb_w[name], colors=yc, txts=classes, name=name)

# with open('data/emb_w.pkl', 'wb') as fw:
#     pickle.dump(emb_w, fw)
with open('data/emb_w.pkl', 'rb') as fr:
    emb_w = pickle.load(fr)

for name in models_list:
    scatter(emb_w[name], colors=yc, txts=cntxts, name=name)

for name in models_list:
    plt.figure(figsize=(16, 16))
    plt.hist(bias_dict[name], bins=256, normed=1, facecolor='b', edgecolor='r', hold=1)
    plt.savefig("data/embedding/bias/%s_b.png" % name)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

emb = tsne_w(weight_dict, "resnet34")
scaler = MinMaxScaler()
emb_range = scaler.fit_transform(emb)
# plt.scatter(*list(emb_range))
from scipy import stats

# stats.ks_2samp(emb_range[:, 0], emb_range[:, 1])
stats.kstest(MinMaxScaler().fit_transform(emb_range[:, 0]), 'uniform')

# stats.kstest(StandardScaler().fit_transform(emb_range[:, 0]), 'norm')



# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# src=Image.open('d:/ex.jpg')
# r,g,b=src.split()
