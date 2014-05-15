# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:35:27 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import numpy as np
from mulm import MUOLSStatsCoefficients

n = 100
px = 10
py_info = 10
py_noize = 90

beta = np.array([2, 1, 0, .5] + [0] * (px - 4)).reshape((px, 1))
title = 'pvals(%i x %i). Significant associations should occur for \
regressor 1, 3 for %i first columns' % (px-1, py_info+py_noize, py_info)

# add intercept
X = np.hstack([np.ones((n, 1)), np.random.randn(n, px-1)])
Y = np.random.randn(n, py_info+py_noize)
# Causal model
Y[:, :py_info] += np.dot(X, beta)

muols = MUOLSStatsCoefficients()
muols.fit(X, Y)
tvals, pvals = muols.stats(X, Y)

import pylab as plt
cax = plt.matshow(pvals[1:,:], cmap=plt.cm.coolwarm_r)
cbar = plt.colorbar(cax)
plt.title(title)
plt.show()
