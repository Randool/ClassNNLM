#!/usr/bin/env python
# encoding: utf-8
import torch


def list2longtensor(x, cuda=True):
    t = torch.tensor(x, dtype=torch.long)
    if cuda:
        t = t.cuda()
    return t


def length2mask(length):
    # mask = torch.ByteTensor(len(length), max(length)).zero_().cuda()
    mask = torch.zeros(len(length), max(length), dtype=torch.bool).cuda()
    for i, l in enumerate(length):
        mask[i, :l].fill_(1)
    return mask


def map_dict_value(func, d):
    d_ = {}
    for k in d:
        d_[k] = func(d[k])
    return d_
