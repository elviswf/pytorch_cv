# -*- coding: utf-8 -*-
"""
@Time    : 2017/11/20 15:10
@Author  : Elvis
"""
"""
 globalConfig.py
  
"""
import torch

# Learning rate parameters
BASE_LR = 0.01
EPOCH_DECAY = 32 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.9 # factor by which the learning rate is reduced.
NUM_EPOCHS = 200

# DATASET INFO
NUM_CLASSES = 200 # set the number of classes in your dataset
# DATA_DIR = "/home/fwu/code/pytorch/data/orbitabase/vhbase_data"
DATA_DIR = "/home/elvis/code/data/cub200/"
BATCH_SIZE = 32

IMAGE_SIZE = 224
MODEL_NAME = "cub_bilinear_f1"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '_' + str(NUM_EPOCHS) + '.pth'
