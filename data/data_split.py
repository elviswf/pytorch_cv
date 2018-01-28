# -*- coding: utf-8 -*-
"""
@Time    : 2017/9/10 14:36
@Author  : Elvis
"""
"""
 data_split.py
  
"""
import os
from sklearn.model_selection import train_test_split
import shutil

src = '/home/fwu/code/pytorch/data/orbitabase/orbital-database-new/trainbase'
# src = '/home/fwu/code/pytorch/data/point'
dst = src + '_data'
train_dir = os.path.join(dst, "train")
val_dir = os.path.join(dst, "val")

classes = os.listdir(src)
for pdir in [train_dir, val_dir]:
    if not os.path.exists(pdir):
        os.makedirs(pdir)

for cls_name in classes:
    train_cls_dir = os.path.join(train_dir, cls_name)
    val_cls_dir = os.path.join(val_dir, cls_name)
    for pdir in [train_cls_dir, val_cls_dir]:
        if not os.path.exists(pdir):
            os.makedirs(pdir)
    class_dir = os.path.join(src, cls_name)
    imgs = os.listdir(class_dir)
    X_train, X_val = train_test_split(imgs, test_size=0.2, random_state=42)
    for img in X_train:
        shutil.copy2(os.path.join(class_dir, img), train_cls_dir)
    for img in X_val:
        shutil.copy2(os.path.join(class_dir, img), val_cls_dir)


predict_dir = "/home/fwu/code/pytorch/data/orbitabase/orbital_new_predict"
for subdir in os.listdir(predict_dir):
    fsubdir = os.path.join(predict_dir, subdir)
    new_dir = os.path.join(fsubdir, "test")
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for file in os.listdir(fsubdir):
        if file.endswith(".png"):
            shutil.copy(os.path.join(fsubdir, file), new_dir)
