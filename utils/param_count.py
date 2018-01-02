# -*- coding: utf-8 -*-
"""
@Time    : 2017/12/3 22:56
@Author  : Elvis
"""
"""
 param_count.py
 
 print(torch_summarize(model))
"""
from torch.nn.modules.module import _addindent
import torch
import numpy as np


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


def lr_scheduler(optimizer, epoch, init_lr=0.01, decay_epoch=0, lr_decay_epoch=6):
    """Decay learning rate by a factor of 0.5 every lr_decay_epoch epochs."""
    dif_epoch = epoch - decay_epoch
    lr = init_lr * (0.8 ** (dif_epoch // lr_decay_epoch))

    if dif_epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer