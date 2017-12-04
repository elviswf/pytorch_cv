# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/4 13:55
@Author  : Elvis
"""
"""
 embed.py
  
"""
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

embed_fp = "log/embed/emb_resnet18_embembedding.pkl"

with open(embed_fp, 'rb') as fr:
    emb_resnet18 = pickle.load(fr)


emb_resnet18.shape
# im = Image.fromarray(emb_resnet18)
# im.save("emb_resnet18.jpg")

import scipy.misc
scipy.misc.imsave('outfile.jpg', emb_resnet18)
import numpy as np
res = emb_resnet18 - np.diag(np.ones(200))
res.sum()

import numpy as np
plt.imshow(emb_resnet18)
plt.savefig("emb_resnet18.png")