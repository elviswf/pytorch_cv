# -*- coding: utf-8 -*-
# file: data_loader.py
# author: JinTian
# time: 10/05/2017 8:53 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import time
import os
from globalConfig import *
from PIL import Image
# import torchsample
import textwrap


class DataLoader(object):
    def __init__(self, data_dir, image_size, batch_size=BATCH_SIZE):
        """
        this class is the normalize data loader of PyTorch.
        The target image size and transforms can edit here.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # torchsample.transforms.Rotate(10),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ]),
        }

        self._init_data_sets()

    def _init_data_sets(self):
        self.data_sets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x])
                          for x in ['train', 'val']}

        self.data_loaders = {'train': torch.utils.data.DataLoader(self.data_sets['train'], batch_size=self.batch_size,
                                                                  shuffle=True, num_workers=4),
                             'val': torch.utils.data.DataLoader(self.data_sets['val'], batch_size=self.batch_size,
                                                                shuffle=False, num_workers=4)}
        self.data_sizes = {x: len(self.data_sets[x]) for x in ['train', 'val']}
        self.data_classes = self.data_sets['train'].classes

    def load_data(self, data_set='train'):
        return self.data_loaders[data_set]

    def show_image(self, tensor, title=None):
        inp = tensor.numpy().transpose((1, 2, 0))
        # put it back as it solved before in transforms
        inp = self.normalize_std * inp + self.normalize_mean
        plt.imshow(inp)
        if title is not None:
            ptitle = "\n".join(textwrap.wrap(",".join(title), 80))
            plt.title(ptitle)
        plt.show()
        # plt.savefig(time.strftime("imgs/%H_%M_%S.jpg", time.localtime()))

    def make_predict_inputs(self, image_file):
        """
        this will make a image to PyTorch inputs, as the same with training images.
        this will return a tensor, default not using CUDA.
        :param image_file:
        :return:
        """
        image = Image.open(image_file)
        image_tensor = self.data_transforms['val'](image).float()
        image_tensor.unsqueeze_(0)
        return Variable(image_tensor)

    def un_normalize(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.normalize_mean, self.normalize_std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor



