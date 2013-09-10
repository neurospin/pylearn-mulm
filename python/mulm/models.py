# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:25:41 2013

@author: ed203246
"""

import scipy
import numpy as np
from sklearn.utils import safe_asarray
from .ols import ols_stats_tcon


class LinearRegression:
    """ Massiv univariate LinearRegression. Repeat linear regression for
    each columns of Y.

    Example
    -------
    >>> import numpy as np
    >>> import mulm
    >>> n_samples = 10
    >>> X = np.random.randn(n_samples, 5)
    >>> X[:, -1] = 1  # Add intercept
    >>> Y = np.random.randn(n_samples, 4)
    >>> betas = np.array([1, 2, 2, 0, 3])
    >>> Y[:, 0] += np.dot(X, betas)
    >>> Y[:, 1] += np.dot(X, betas)
    >>> linreg = mulm.LinearRegression()
    >>> linreg.fit(X, Y)
    >>> Ypred = linreg.predict(X)
    >>> ss_errors = np.sum((Y - Ypred) ** 2, axis=0)
    """
    def __init__(self, **kwargs):
        self.coef_ = None

    def fit(self, X, Y):
        X = safe_asarray(X)
        Y = safe_asarray(Y)
        self.coef_ = np.dot(scipy.linalg.pinv(X), Y)
        return self

    def predict(self, X):
        X = safe_asarray(X)
        return np.dot(X, self.coef_)

    def stats(self, X, Y, contrast, pval=True):
        Ypred = self.predict(X)
        ss_errors = np.sum((Y - Ypred) ** 2, axis=0)
        tval, pvalt = ols_stats_tcon(X, self.coef_, ss_errors, contrast,
                                          pval)
        return tval, pvalt