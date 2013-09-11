# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 19:37:43 2013

@author: edouard.duchesnay@cea.fr
"""

from scipy import stats
import scipy
import numpy as np
from sklearn.utils import array2d


def ols(X, Y):
    """ Massively application of Ordinary Least Square for each column of Y    
    """
    betas = np.dot(scipy.linalg.pinv(X), Y)
    ss_errors = np.sum((Y - np.dot(X, betas)) ** 2, axis=0)
    return betas, ss_errors


