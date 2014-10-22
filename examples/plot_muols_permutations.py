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
title = 'Empirical pvalues'

# add intercept
X = np.hstack([np.ones((n, 1)), np.random.randn(n, px-1)])
Y = np.random.randn(n, py_info+py_noize)
# Causal model
Y[:, :py_info] += np.dot(X, beta)

muols = MUOLSStatsCoefficients()
muols.fit(X, Y)
contrast = [0, 1] + [0] * (X.shape[1] - 2)
tvals, pval, _ = muols.stats_t_permutations(X=X, Y=Y, contrast=contrast, nperms=1000)

import pylab as plt
plt.plot(pval)#, cmap=plt.cm.coolwarm_r)
plt.title(title)
plt.show()

