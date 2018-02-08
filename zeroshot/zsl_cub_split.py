# -*- coding: utf-8 -*-
"""
@Time    : 2018/1/24 18:45
@Author  : Elvis

zsl_split.py
split 150 -> 50
"""
import os
import shutil

origin_dir = "/home/elvis/data/attribute/CUB_200_2011"

train_file = "/home/elvis/data/attribute/CUB_200_2011/zsl/trainvalclasses.txt"
test_file = "/home/elvis/data/attribute/CUB_200_2011/zsl/testclasses.txt"

train_classes = []
test_classes = []
with open(train_file, "r") as fr:
    for line in fr.readlines():
        train_classes.append(line.strip())

with open(test_file, "r") as fr:
    for line in fr.readlines():
        test_classes.append(line.strip())

img_dir = os.path.join(origin_dir, "images")
zsl_train_dir = os.path.join(origin_dir, "zsl", "zsl_train")
zsl_test_dir = os.path.join(origin_dir, "zsl", "zsl_test")

for ci in train_classes:
    src_dir = os.path.join(img_dir, ci)
    shutil.move(src_dir, zsl_train_dir)

for ci in test_classes:
    src_dir = os.path.join(img_dir, ci)
    shutil.move(src_dir, zsl_test_dir)


from sklearn.model_selection import train_test_split
zsl_trainval_dir = os.path.join(origin_dir, "zsl", "trainval")
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
    X_train, X_val = train_test_split(imgs, test_size=0.2, random_state=42)
    for img in X_train:
        shutil.copy2(os.path.join(class_dir, img), train_cls_dir)
    for img in X_val:
        shutil.copy2(os.path.join(class_dir, img), val_cls_dir)


import numpy as np
import pickle
cub_attr = np.loadtxt("/home/elvis/data/attribute/CUB_200_2011/zsl/class_attribute_labels_continuous.txt")  # ci order

np.save("data/cub_attr.npy", cub_attr)
order_classes = sorted(train_classes) + sorted(test_classes)
order_id = [int(name.split(".")[0]) - 1 for name in order_classes]  # id -> ci
order_cub_attr = cub_attr[order_id]
order_cub_attr.shape
np.save("data/order_cub_attr.npy", order_cub_attr)
# w_attr = np.load("data/w_attr.npy")
# w_attr[5] == cub_attr[10]

order_train_file = os.path.join(origin_dir, "zsl", "order_train_classes.txt")
order_test_file = os.path.join(origin_dir, "zsl", "order_test_classes.txt")
order_classes_file = os.path.join(origin_dir, "zsl", "order_classes.txt")

with open(order_train_file, "w") as fw:
    fw.writelines("\n".join(train_classes))

with open(order_test_file, "w") as fw:
    fw.writelines("\n".join(test_classes))

with open(order_classes_file, "w") as fw:
    fw.writelines("\n".join(order_classes))


"""
cub gzsl rename directory
"""
gzsl_dir = "/home/elvis/data/attribute/CUB_200_2011/zsl/gzsl_test/val"

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
    new_name = ("%03d" % c_ri) + "." + c_fi.split(".")[1]
    os.rename(os.path.join(gzsl_dir, c_fi), os.path.join(gzsl_dir, new_name))


cub_attr = np.load("data/order_cub_attr.npy")
cub_attr.shape
cub_attr_sum = np.sum(cub_attr, axis=0)
cub_attr1 = cub_attr / cub_attr_sum
cub_attr1.shape
cub_attr[10, 21] / cub_attr_sum[21]
cub_attr1[10, 21]
np.save("data/order_cub_attr1.npy", cub_attr1)
