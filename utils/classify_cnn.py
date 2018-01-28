# -*- coding: utf-8 -*-
"""
@Time    : 2017/11/5 23:34
@Author  : Elvis
"""
"""
 classify_cnn.py
  
conda config --set auto_update_conda false
"""
import matplotlib

matplotlib.use("Agg")
import skimage.io
import os
from matplotlib import pyplot as plt

file_name = '/home/fwu/code/pytorch/data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
# 26132.jpg
if not os.access(file_name, os.R_OK):
    file_URL = 'http://www.zooclub.ru/attach/26000/26132.jpg'
    os.system('wget ' + file_URL)

img = skimage.io.imread(file_name)
# plt.imshow(img)
# plt.axis('off')
# plt.show()

import torchvision

# get model
resnet_18 = torchvision.models.resnet18(pretrained=True)
resnet_18.eval()

# get classes
file_name = 'synset_words.txt'
if not os.access(file_name, os.W_OK):
    synset_URL = 'https://github.com/szagoruyko/functional-zoo/raw/master/synset_words.txt'
    os.system('wget ' + synset_URL)

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ', 1)[1].split(', ', 1)[0])
classes = tuple(classes)

from torchvision import transforms

centre_crop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from torch.autograd import Variable
from torch.nn import functional as F

file_name = '/home/fwu/code/pytorch/data/CUB_200_2011/images/050.Eared_Grebe/Eared_Grebe_0013_34150.jpg'
img = skimage.io.imread(file_name)
# get top 5 probabilities
x = Variable(centre_crop(img).unsqueeze(0), volatile=True)
logit = resnet_18(x)
h_x = F.softmax(logit).data.squeeze()
probs, idx = h_x.sort(0, True)
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
