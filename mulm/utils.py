# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:46:38 2013

@author: jinpeng.li@cea.fr
"""
import numpy as np


def variance(list):
    avg = np.mean(list)
    sq_diff = 0
    for elem in list:
        sq_diff += (elem - avg) ** 2
    return sq_diff / len(list)


if __name__ == "__main__":
    print variance([1, 2, 3])
