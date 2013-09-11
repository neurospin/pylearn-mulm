# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:25:41 2013

@author: ed203246
"""

import scipy
import numpy as np
from sklearn.utils import safe_asarray
from mulm.ols import ols_stats_tcon
from mulm.ols import ols_stats_fcon
from mulm.ols import ols


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


class OLSRegression:
    def __init__(self):
        pass

    def transform(self, Y, y_group_indices, X, x_group_indices):
        """

        Example
        -------
        import numpy as np
        import random
        from mulm.models import OLSRegression
        n_samples = 10
        n_xfeatures = 20
        n_yfeatures = 15
        x_n_groups = 3
        y_n_groups = 2

        X = np.random.randn(n_samples, n_xfeatures)
        Y = np.random.randn(n_samples, n_yfeatures)
        x_group_indices = np.array([randint(0, x_n_groups)\
            for i in xrange(n_xfeatures)])
        y_group_indices = np.array([randint(0, y_n_groups)\
            for i in xrange(n_yfeatures)])

        regression = OLSRegression()
        print regression.transform(Y, y_group_indices, X, x_group_indices)

        """
        ret = {}
        y_uni_group_indices = set(y_group_indices)
        x_uni_group_indices = set(x_group_indices)
        for y_uni_group_index in y_uni_group_indices:
            for x_uni_group_index in x_uni_group_indices:
                if y_uni_group_index == -1 or x_uni_group_index == -1:
                    continue
                # y_uni_group_index = y_uni_group_indices.pop()
                # x_uni_group_index = x_uni_group_indices.pop()
                gY = Y[:, y_group_indices == y_uni_group_index]
                gX = X[:, x_group_indices == x_uni_group_index]
                betas, ss_errors = ols(gX, gY)
                contrast = np.zeros(gX.shape[1])
                contrast[0] = 1
                tp, t = ols_stats_tcon(gX, betas, ss_errors,\
                    contrast=contrast, pval=True)
                fp, f = ols_stats_fcon(gX, betas, ss_errors,\
                    contrast=contrast, pval=True)
                key = "gy_%s_gx_%s" % (y_uni_group_index, x_uni_group_index)
                key_t = key + "_t"
                key_tp = key + "_tp"
                key_f = key + "_f"
                key_fp = key + "_fp"
                ret[key_t] = t
                ret[key_tp] = tp
                ret[key_f] = f
                ret[key_fp] = fp
        return ret
