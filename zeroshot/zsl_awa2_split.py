# -*- coding: utf-8 -*-
"""
@Time    : 2018/1/24 18:45
@Author  : Elvis

zsl_split.py
split 150 -> 50
"""
import os
import shutil

origin_dir = "/home/elvis/data/attribute/AwA/Animals_with_Attributes2"

train_file = os.path.join(origin_dir, "trainclasses.txt")
test_file = os.path.join(origin_dir, "testclasses.txt")

train_classes = []
test_classes = []
with open(train_file, "r") as fr:
    for line in fr.readlines():
        train_classes.append(line.strip())

with open(test_file, "r") as fr:
    for line in fr.readlines():
        test_classes.append(line.strip())

img_dir = os.path.join(origin_dir, "JPEGImages")
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
attr_path = os.path.join(origin_dir, "predicate-matrix-continuous.txt")
awa2_attr = np.loadtxt(attr_path)  # ci order
awa2_attr.shape

allclass_fp = os.path.join(origin_dir, "classes.txt")
allclasses = []
with open(allclass_fp, "r") as fr:
    for line in fr.readlines():
        allclasses.append(line.strip())


# class_id = [cid for cid, name in enumerate(allclasses)]  # id -> ci
class_to_idx = {name: idx for idx, name in enumerate(allclasses)}  # ci -> id
# class_id[:6]

order_classes = sorted(train_classes) + sorted(test_classes)
order_classes
order_id = [class_to_idx[ci] for ci in order_classes]
order_awa2_attr = awa2_attr[order_id]
np.save("data/order_awa2_attr.npy", order_awa2_attr)  # (50, 85)
# w_attr[5] == cub_attr[10]

order_train_file = os.path.join(origin_dir, "order_train_classes.txt")
order_test_file = os.path.join(origin_dir, "order_test_classes.txt")
order_classes_file = os.path.join(origin_dir, "order_classes.txt")

with open(order_train_file, "w") as fw:
    fw.writelines("\n".join(train_classes))

with open(order_test_file, "w") as fw:
    fw.writelines("\n".join(test_classes))

with open(order_classes_file, "w") as fw:
    fw.writelines("\n".join(order_classes))


