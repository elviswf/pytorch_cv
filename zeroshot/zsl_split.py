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

allclass_fp = os.path.join(origin_dir, "zsl", "allclasses.txt")
allclasses = []
with open(allclass_fp, "r") as fr:
    for line in fr.readlines():
        allclasses.append(line.strip())


class_id = [int(name.split(".")[0]) - 1 for name in allclasses]  # id -> ci
# class_to_idx = {ci: idx for idx, ci in enumerate(class_id)}  # ci -> id
# class_id[:6]
w_attr = cub_attr[class_id]

np.save("data/w_attr.npy", w_attr)
# w_attr[5] == cub_attr[10]




