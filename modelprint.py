# -*- coding: utf-8 -*-
"""
@Time    : 2018/2/5 14:28
@Author  : Elvis
 modelprint.py
  
"""
import torch

MODEL_NAME = "gzsl_resnet50_g_g17"
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

print("Model: " + MODEL_NAME)
print("==> Resuming from checkpoint...")
checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
net = checkpoint["net"]
best_acc = checkpoint["acc"]
start_epoch = checkpoint["epoch"]
optimizer = checkpoint["optimizer"]

print(net)
print("best_acc: %f" % best_acc)
print("start_epoch: %f" % start_epoch)
# print("optimizer: %f" % optimizer)

fc0 = net.fc0
fc1 = net.fc1
fc0_w = fc1[0].weight.data
fc0_b = fc1[0].bias.data

import matplotlib.pyplot as plt
# matplotlib.use("Agg")
import skimage.io

# file_name = '/home/elvis/data/attribute/CUB_200_2011/zsl/gzsl_test/val/061.Green_Jay/Green_Jay_0090_65895.jpg'
file_name = "/home/elvis/data/attribute/CUB_200_2011/zsl/gzsl_test/val/200.Common_Yellowthroat/Common_Yellowthroat_0003_190521.jpg"
img = skimage.io.imread(file_name)
plt.ion()
plt.imshow(img)
plt.savefig()

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

net._modules.keys()
fc1_layer = net._modules.get('fc1')
fc2_layer = net._modules.get('fc2')
wt_list = []
xt_list = []


def fc1_fun(m, i, o):
    wt_list.append(o.data.cpu().numpy())
    print(
        'm:', type(m),
        '\ni:', type(i),
            '\n   len:', len(i),
            '\n   type:', type(i[0]),
            '\n   data size:', i[0].data.size(),
            '\n   data type:', i[0].data.type(),
        '\no:', type(o),
            '\n   data size:', o.data.size(),
            '\n   data type:', o.data.type(),
    )


def fc2_fun(m, i, o):
    xt_list.append(i[0].data.cpu().numpy())


fc1_h = fc1_layer.register_forward_hook(fc1_fun)
fc2_h = fc2_layer.register_forward_hook(fc2_fun)
# fc1_h.remove()
# fc2_h.remove()

# get top 5 probabilities
import numpy as np
x = Variable(centre_crop(img).unsqueeze(0), volatile=True).cuda()
logit, attr = net(x)
len(wt_list)
np.save("data/output/wt_g17.npy", wt_list)
np.save("data/output/xt_g17.npy", xt_list)

x_attr = attr.cpu().data.numpy()
np.save("data/output/x_attr_g17.npy", x_attr)

x_attr.shape

h_x = F.softmax(logit).data.squeeze()
probs, idx = h_x.sort(0, True)
idx[:9]


"""
test images
"""
from torchvision import datasets
import os
root_dir = "/home/elvis/data/attribute/CUB_200_2011/zsl/gzsl_test/val"
BATCH_SIZE = 32


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
        self.val_set = datasets.ImageFolder(root_dir, self.data_transforms['val'])
        self.val_loaders = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=4)
        self.data_sizes = len(self.val_set)
        self.data_classes = self.val_set.classes

    def load_data(self, data_set='train'):
        return self.data_loaders[data_set]

    # def show_image(self, tensor, title=None):
    #     inp = tensor.numpy().transpose((1, 2, 0))
    #     # put it back as it solved before in transforms
    #     inp = self.normalize_std * inp + self.normalize_mean
    #     plt.imshow(inp)
    #     if title is not None:
    #         ptitle = "\n".join(textwrap.wrap(",".join(title), 80))
    #         plt.title(ptitle)
    #     plt.show()
    #     # plt.savefig(time.strftime("imgs/%H_%M_%S.jpg", time.localtime()))
    #
    # def make_predict_inputs(self, image_file):
    #     """
    #     this will make a image to PyTorch inputs, as the same with training images.
    #     this will return a tensor, default not using CUDA.
    #     :param image_file:
    #     :return:
    #     """
    #     image = Image.open(image_file)
    #     image_tensor = self.data_transforms['val'](image).float()
    #     image_tensor.unsqueeze_(0)
    #     return Variable(image_tensor)
    #
    # def un_normalize(self, tensor):
    #     """
    #     Args:
    #         tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    #     Returns:
    #         Tensor: Normalized image.
    #     """
    #     for t, m, s in zip(tensor, self.normalize_mean, self.normalize_std):
    #         t.mul_(s).add_(m)
    #         # The normalize code -> t.sub_(m).div_(s)
    #     return tensor



