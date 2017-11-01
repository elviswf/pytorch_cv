# -*- coding: utf-8 -*-
"""
@Time    : 2017/2/24 10:13
@Author  : Elvis
"""
"""
 alexnet_trans.py
  
"""
import numpy as np
from sklearn.manifold import TSNE
# Random state.
RS = 20171106
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.preprocessing import normalize
from PIL import Image
import seaborn as sns
import torch
from torchvision import models


def getW(name="alexnet"):
    model = models.__dict__[name](pretrained=True)
    # model.


models_list = ['alexnet', 'inception_v3', 'resnet18', 'resnet34', 'resnet50',
               'resnet101', 'resnet152', 'vgg16', 'vgg19']


sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

Xw = np.load("inceptResv2W.npy")  ##alexW
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
emb = model.fit_transform(Xw)
y = np.array(list(range(emb.shape[0])))
txts = open("synset_words_str.txt").readlines()


# Scale and visualize the embedding vectors
def scatter(x, colors, txts=None, title=None, ex=None):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", colors.size))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.title(title)
    if ex is not None:
        sc = ax.scatter(ex[:, 0], ex[:, 1], lw=0, s=40)
        # We add the labels for each digit.

    for i in range(colors.size):
        # Position of each label.
        # if (i % 8 == 0):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        if (len(txts[i]) > 10):
            txts[i] = txts[i][:15]
        txt = ax.text(xtext, ytext, txts[i], fontsize=8)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=3, foreground="w"),
        PathEffects.Normal()])

    return f, ax, sc


# img = Image.fromarray(Xw.transpose(), 'RGB')
# img.save('w8n.png')
# img.show()

scatter(emb, colors=y, txts=txts, title="inceptResv2W 1001*536 embedding")
# plt.imshow(X.transpose())
#########################################################################
xmat = np.load("carfeat4096.npy")

xmat.shape
modelcar = TSNE(n_components=2, random_state=0)
embX = modelcar.fit_transform(xmat.transpose())
scatter(emb, colors=y, txts=txts, title="AlexNet 1000*4096 embedding", ex=embX)
car_labels = open("train_perfect_preds.txt").readlines()
colors = [(int(ci) - 1) for ci in car_labels]
colors = colors[:1000]
colors = np.array(colors)
embX
scatter(embX, colors=colors, txts=car_labels, title="Cars on AlexNet 1000*4096 embedding")

#########################################################################
cars1000 = np.load("carfeat1000.npy")
cars1000.shape
modelcar = TSNE(n_components=2, random_state=0)
embX = modelcar.fit_transform(cars1000)
car_labels = open("train_perfect_preds.txt").readlines()
colors = [(int(ci) - 1) for ci in car_labels]
colors = np.array(colors)
embX.shape
palette = np.array(sns.color_palette("hls", 196))
# We create a scatter plot.
f = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')
sc = ax.scatter(embX[:, 0], embX[:, 1], lw=0, s=40,
                c=palette[colors.astype(np.int)])
plt.xlim(-25, 25)
plt.ylim(-25, 25)
ax.axis('off')
ax.axis('tight')
plt.title("Cars on AlexNet 1000 embedding")
#########################################################################
Xb = np.load("alexB.npy")
Xb.shape
imgb = Image.fromarray(Xb.transpose())
imgb.imshow(Xb)

# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# src=Image.open('d:/ex.jpg')
# r,g,b=src.split()


plt.hist(Xb, bins=256, normed=1, facecolor='b', edgecolor='r', hold=1)

plt.rcdefaults()
import pandas as pd

method = (str(bl) for bl in range(1000))
bData = pd.Series(Xb, index=method)
bData.plot(kind='bar', color='b', alpha=0.7)
plt.show()

feats = [str(f) for f in range(4096)]
wData = pd.DataFrame(Xw, columns=feats)
wData.head()
for feat in feats:
    wData[feat]

ax = plt.figure()
w1d = wData["1"]
w1d.plot(kind='bar', color='b', alpha=0.7)
plt.show()


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
[b, c, d] = plt.plot(x, softmax(scores).T, linewidth=2)
lg = plt.legend([b, c, d], ["x", "1.0", "0.2"], loc=1)
lg.draw_frame = True
plt.show()

from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'
pickle_file1 = 'notMNIST2.pickle'
fw = open(pickle_file1, 'wb')
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    pickle.dump(save, fw, 2)
fw.close()
