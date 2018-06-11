# -*- coding: utf-8 -*-
"""
@Time    : 2018/3/9 21:26
@Author  : Elvis
"""
"""
 tsne_model.py
  
"""
import torch
from torch import nn
from torchvision.models import resnet50
import numpy as np


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
RS = 20180310


def tsne_w(Xw):
    model = TSNE(n_components=2, random_state=RS)
    np.set_printoptions(suppress=True)
    emb = model.fit_transform(Xw)
    return emb


# Scale and visualize the embedding vectors
def scatter(x, colors, txts=None, save_name="resnet50_fc_no_bias.pdf", font=6):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", colors.size))

    # We create a scatter plot.
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(aspect='equal')
    ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.title("2D embedding")

    for i in range(colors.size):
        # Position of each label.
        # if (i % 8 == 0):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        if len(txts[i]) > 10:
            txts[i] = txts[i][:8]
        txt = ax.text(xtext, ytext, txts[i], fontsize=font)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=3, foreground="w"),
        PathEffects.Normal()])

    plt.savefig("data/embedding/" + save_name)
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
        classes.append(line.strip().split(' ', 1)[1])

emb_w = tsne_w(w_np)
emb_w.shape

from sklearn.cluster import KMeans

kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
kmeans.fit(emb_w)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = emb_w[:, 0].min() - 1, emb_w[:, 0].max() + 1
y_min, y_max = emb_w[:, 1].min() - 1, emb_w[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 12))
plt.clf()
ax = plt.subplot(aspect='equal')
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel1,
           aspect='auto', origin='lower')

palette = np.array(sns.color_palette("hls", nclass))
# plt.plot(emb_w[:, 0], emb_w[:, 1], 'k.', markersize=2)
ax.scatter(emb_w[:, 0], emb_w[:, 1], lw=0, s=20, c=palette[yc.astype(np.int)])
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
lebels = kmeans.labels_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)

txts = classes
for i in range(0, nclass, 6):
    # Position of each label.
    # if (i % 8 == 0):
    xtext, ytext = emb_w[i]
    if len(txts[i]) > 10:
        txts[i] = txts[i][:28]
    txt = ax.text(xtext, ytext, txts[i], fontsize=8)
# plt.title('K-means clustering on fc_W in resnet50')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.savefig("data/embedding/imagenet1k_w1.pdf")

scatter(emb_w, colors=yc, txts=classes, save_name="resnet50_fc_no_bias3.pdf")

"""
mnist
"""
MODEL_SAVE_FILE = "mnist_conv2w.pth"
print("==> Resuming from checkpoint...")
checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
net = checkpoint["net"]
best_acc = checkpoint["acc"]

print(net)
w_fc = net.classifier[3].weight
w_mnist = w_fc.data.cpu().numpy()
w_mnist.shape
np.save("data/w_mnist.npy", w_mnist)

nclass = 10
yc = np.array(list(range(nclass)))
classes = [str(i) for i in range(10)]
emb_w_mnist = tsne_w(w_mnist)
scatter(emb_w_mnist, colors=yc, txts=classes, save_name="mnist_w.pdf", font=12)

"""
fashion-mnist
"""
MODEL_SAVE_FILE = "fmnist_conv2w.pth"
print("==> Resuming from checkpoint...")
checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
net = checkpoint["net"]
best_acc = checkpoint["acc"]

print(net)
w_fc = net.classifier[3].weight
w_fmnist = w_fc.data.cpu().numpy()
w_fmnist.shape
np.save("data/w_fmnist.npy", w_fmnist)

w_fmnist = np.load("data/w_fmnist.npy")
nclass = 10
yc = np.array(list(range(nclass)))
fmnist_path = "data/fashion_mnist_classes.txt"
classes = list()
with open(fmnist_path) as class_file:
    for line in class_file:
        classes.append(line.strip().split('\t', 1)[1])

emb_w_fmnist = tsne_w(w_fmnist)


from sklearn.cluster import KMeans

kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10)
kmeans.fit(emb_w_fmnist)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .2  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = emb_w_fmnist[:, 0].min() - 1, emb_w_fmnist[:, 0].max() + 1
y_min, y_max = emb_w_fmnist[:, 1].min() - 1, emb_w_fmnist[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
# figsize=(12, 12)
plt.figure(1)
plt.clf()
ax = plt.subplot(aspect='equal')
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel1,
           aspect='auto', origin='lower')

palette = np.array(sns.color_palette("hls", nclass))
# plt.plot(emb_w_fmnist[:, 0], emb_w_fmnist[:, 1], 'k.', markersize=2)
ax.scatter(emb_w_fmnist[:, 0], emb_w_fmnist[:, 1], lw=0, s=20, c=palette[yc.astype(np.int)])
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
lebels = kmeans.labels_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)

txts = classes
for i in range(nclass):
    xtext, ytext = emb_w_fmnist[i]
    txt = ax.text(xtext*0.9-10, ytext, txts[i], fontsize=8)
# plt.title('K-means clustering on fc_W in resnet50')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.savefig("data/embedding/emb_w_fmnist.pdf")



scatter(emb_w_fmnist, colors=yc, txts=classes, save_name="fmnist_w.pdf", font=12)

# plt.imshow(fm_x_train[0].reshape((28, 28)), cmap='gray')


"""
cifar10
"""


def unpickle(file):  # 该函数将cifar10提供的文件读取到python的数据结构(字典)中
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='iso-8859-1')
    fo.close()
    return dict


MODEL_SAVE_FILE = "cifar10_resnet18w.pth"
print("==> Resuming from checkpoint...")
checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
net = checkpoint["net"]
best_acc = checkpoint["acc"]

print(net)
w_fc = net.fc[3].weight
w_cifar10 = w_fc.data.cpu().numpy()
w_cifar10.shape
np.save("data/w_cifar10.npy", w_cifar10)

nclass = 10
yc = np.array(list(range(nclass)))
cifar10_path = "/home/elvis/code/data/cifar/cifar-10-batches-py/batches.meta"
cifar10_meta = unpickle(cifar10_path)
cifar10_classes = cifar10_meta['label_names']
print(cifar10_classes)

emb_w_cifar10 = tsne_w(w_cifar10)
scatter(emb_w_cifar10, colors=yc, txts=cifar10_classes, save_name="cifar10_w.pdf", font=12)

"""
cifar100
"""
MODEL_SAVE_FILE = "cifar100_resnet18w.pth"
print("==> Resuming from checkpoint...")
checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
net = checkpoint["net"]
best_acc = checkpoint["acc"]

print(net)
w_fc = net.fc[3].weight
w_cifar100 = w_fc.data.cpu().numpy()
w_cifar100.shape
np.save("data/w_cifar100.npy", w_cifar100)

nclass = 100
yc = np.array(list(range(nclass)))

cifar100_path = "/home/elvis/code/data/cifar/cifar-100-python/meta"
cifar100_meta = unpickle(cifar100_path)
cifar100_classes = cifar100_meta['fine_label_names']
len(cifar100_classes)

emb_w_cifar100 = tsne_w(w_cifar100)
scatter(emb_w_cifar100, colors=yc, txts=cifar100_classes, save_name="cifar100_w.pdf", font=12)


