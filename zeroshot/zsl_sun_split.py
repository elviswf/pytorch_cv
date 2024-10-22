# -*- coding: utf-8 -*-
"""
@Time    : 2018/1/24 18:45
@Author  : Elvis

zsl_split.py
split 150 -> 50
"""
import os
import shutil

origin_dir = "/home/elvis/data/attribute/SUN"

train_file = os.path.join(origin_dir, "trainvalclasses.txt")
test_file = os.path.join(origin_dir, "testclasses.txt")

train_classes = []
test_classes = []
with open(train_file, "r") as fr:
    for line in fr.readlines():
        train_classes.append(line.strip())

with open(test_file, "r") as fr:
    for line in fr.readlines():
        test_classes.append(line.strip())

len(test_classes)

img_dir = os.path.join(origin_dir, "images")
zsl_train_dir = os.path.join(origin_dir, "zsl", "zsl_train")
zsl_test_dir = os.path.join(origin_dir, "zsl", "zsl_test")

for ci in train_classes:
    src_dir = os.path.join(img_dir, ci[0], ci)
    if not os.path.exists(src_dir):
        c1, c2 = ci.split("_", 1)
        src_dir = os.path.join(img_dir, ci[0], c1, c2)
        if not os.path.exists(src_dir):
            cn = ci.split("_")
            c1 = "_".join(cn[:-1])
            src_dir = os.path.join(img_dir, ci[0], c1, cn[-1])
            if not os.path.exists(src_dir):
                cn = ci.split("_")
                c1 = "_".join(cn[:-2])
                c2 = "_".join(cn[-2:])
                src_dir = os.path.join(img_dir, ci[0], c1, c2)
    shutil.copytree(src_dir, os.path.join(zsl_train_dir, ci))

for ci in test_classes:
    src_dir = os.path.join(img_dir, ci[0], ci)
    if not os.path.exists(src_dir):
        c1, c2 = ci.split("_", 1)
        src_dir = os.path.join(img_dir, ci[0], c1, c2)
        if not os.path.exists(src_dir):
            cn = ci.split("_")
            c1 = "_".join(cn[:-1])
            src_dir = os.path.join(img_dir, ci[0], c1, cn[-1])
            if not os.path.exists(src_dir):
                cn = ci.split("_")
                c1 = "_".join(cn[:-2])
                c2 = "_".join(cn[-2:])
                src_dir = os.path.join(img_dir, ci[0], c1, c2)
    shutil.copytree(src_dir, os.path.join(zsl_test_dir, ci))

from sklearn.model_selection import train_test_split
zsl_trainval_dir = os.path.join(origin_dir, "zsl", "trainval1")
train_dir = os.path.join(zsl_trainval_dir, "train")
val_dir = os.path.join(zsl_trainval_dir, "val")

for pdir in [train_dir, val_dir]:
    if not os.path.exists(pdir):
        os.makedirs(pdir)

for cls_name in train_classes:
    train_cls_dir = os.path.join(train_dir, cls_name)
    val_cls_dir = os.path.join(val_dir, cls_name)
    for pdir in [train_cls_dir, val_cls_dir]:
        if not os.path.exists(pdir):
            os.makedirs(pdir)
    class_dir = os.path.join(zsl_train_dir, cls_name)
    imgs = os.listdir(class_dir)
    X_train, X_val = train_test_split(imgs, test_size=0.3, random_state=42)
    for img in X_train:
        shutil.copy2(os.path.join(class_dir, img), train_cls_dir)
    for img in X_val:
        shutil.copy2(os.path.join(class_dir, img), val_cls_dir)


import numpy as np
import pickle
sun_attr = np.loadtxt("/home/elvis/data/attribute/SUN/sun_class_attr.txt")  # ci order
sun_class_file = os.path.join(origin_dir, "sun_classes_names.txt")
class_names = []
with open(sun_class_file, "r") as fr:
    for line in fr.readlines():
        class_names.append(line.strip())

sun_attr.shape
np.save("data/sun_attr.npy", sun_attr)
order_classes = sorted(train_classes) + sorted(test_classes)
order_id = [class_names.index(name) for name in order_classes]  # id -> ci
order_sun_attr = sun_attr[order_id]
order_sun_attr.shape
np.save("data/order_sun_attr.npy", order_sun_attr)
# w_attr = np.load("data/w_attr.npy")
# w_attr[5] == cub_attr[10]

order_train_file = os.path.join(origin_dir, "zsl", "order_train_classes.txt")
order_test_file = os.path.join(origin_dir, "zsl", "order_test_classes.txt")
order_classes_file = os.path.join(origin_dir, "zsl", "order_classes.txt")

with open(order_train_file, "w") as fw:
    fw.writelines("\n".join(sorted(train_classes)))

with open(order_test_file, "w") as fw:
    fw.writelines("\n".join(sorted(test_classes)))

with open(order_classes_file, "w") as fw:
    fw.writelines("\n".join(order_classes))


"""
cub gzsl rename directory
"""
gzsl_dir = "/home/elvis/data/attribute/SUN/zsl/gzsl_test/val"
shutil.copytree(val_dir, gzsl_dir)

order_classes = []
order_classes_file = os.path.join(origin_dir, "zsl", "order_classes.txt")
with open(order_classes_file, "r") as fr:
    for line in fr.readlines():
        order_classes.append(line.strip())

classes_to_rank = dict()
for ri, ci in enumerate(order_classes):
    classes_to_rank[ci] = ri

for c_fi in os.listdir(gzsl_dir):
    c_ri = classes_to_rank[c_fi] + 1
    new_name = ("%03d" % c_ri) + "." + c_fi
    os.rename(os.path.join(gzsl_dir, c_fi), os.path.join(gzsl_dir, new_name))


cub_attr = np.load("data/order_sun_attr.npy")
cub_attr_sum = np.sum(cub_attr, axis=1)
cub_attr_sum.shape
cub_attr1 = cub_attr.T / cub_attr_sum
cub_attr1.shape
cub_attr1 = cub_attr1.T
len(np.sum(cub_attr1, 1))
np.save("data/order_cub_attr1.npy", cub_attr1)
cub_attr1 = np.load("data/order_cub_attr1.npy")
cub_attr1 = cub_attr1 * 10
len(np.sum(cub_attr1, 1))

"""
pairwise_distances matrix  label propagation
"""
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
w = np.load("data/order_sun_attr.npy")
w[1]
wd = pairwise_distances(w, metric="euclidean")
num = w.shape[0]
ws = np.diag(np.ones(num))

beta = 1.4
for i in range(num):
    for j in range(i):
        ws[i, j] = np.exp(-beta * wd[i, j]**2 / (np.partition(wd[i, :], 1)[1] * np.partition(wd[j, :], 1)[1]))
        if ws[i, j] < 0.05:
            ws[i, j] = 0.
        ws[j, i] = ws[i, j]

ws_p = ws / np.sum(ws, axis=0)
ws_p.diagonal()
np.save("data/sun_ws_14.npy", ws_p)
