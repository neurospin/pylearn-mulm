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


class MUOLS:
    """Mass-univariate linear modeling based Ordinary Least Squares.
    Fit independant OLS models for each columns of Y.

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
        tval, pvalt = self.ols_stats_tcon(X, ss_errors, contrast,
                                          pval)
        return tval, pvalt

    def t_stats(self, X, Y, contrast, pval=False):
        """Compute statistics (t-scores and p-value associated to contrast)

        Parameters
        ----------

        X 2-d array

        betas  2-d array

        ss_errors  2-d array

        contrast  1-d array

        pval: boolean

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
        >>> 
        >>> betas, ss_errors = mulm.ols(X, Y)
        >>> p, t = mulm.ols_stats_tcon(X, betas, ss_errors, contrast=[1, 0, 0, 0, 0], pval=True)
        >>> p, f = mulm.ols_stats_fcon(X, betas, ss_errors, contrast=[1, 0, 0, 0, 0], pval=True)
        """
        Ypred = self.predict(X)
        betas = self.coef_
        ss_errors = np.sum((Y - Ypred) ** 2, axis=0)
        ccontrast = np.asarray(contrast)
        n = X.shape[0]
        # t = c'beta / std(c'beta)
        # std(c'beta) = sqrt(var_err (c'X+)(X+'c))
        Xpinv = scipy.linalg.pinv(X)
        cXpinv = np.dot(ccontrast, Xpinv)
        R = np.eye(n) - np.dot(X, Xpinv)
        df = np.trace(R)
        ## Broadcast over ss errors
        var_errors = ss_errors / df
        std_cbeta = np.sqrt(var_errors * np.dot(cXpinv, cXpinv.T))
        t_stats = np.dot(ccontrast, betas) / std_cbeta
        if not pval:
            return (t_stats, None)
        else:
            p_vals = stats.t.sf(t_stats, df)
            return t_stats, p_vals


    def f_stats(self, X, Y, contrast, pval=False):
        Ypred = self.predict(X)
        betas = self.coef_
        ss_errors = np.sum((Y - Ypred) ** 2, axis=0)
        C1 = array2d(contrast).T
        n = X.shape[0]
        p = X.shape[1]
        Xpinv = scipy.linalg.pinv(X)
        rank_x = np.linalg.matrix_rank(Xpinv)
        C0 = np.eye(p) - np.dot(C1, scipy.linalg.pinv(C1))  # Ortho. cont. to C1
        X0 = np.dot(X, C0)  # Design matrix of the reduced model
        X0pinv = scipy.linalg.pinv(X0)
        rank_x0 = np.linalg.matrix_rank(X0pinv)
        # Find the subspace (X1) of Xc1, which is orthogonal to X0
        # The projection matrix M due to X1 can be derived from the residual
        # forming matrix of the reduced model X0
        # R0 is the residual forming matrix of the reduced model
        R0 = np.eye(n) - np.dot(X0, X0pinv)
        # R is the residual forming matrix of the full model
        R = np.eye(n) - np.dot(X, Xpinv)
        # compute the projection matrix
        M = R0 - R
        Ypred = np.dot(X, betas)
        SS = np.sum(Ypred * np.dot(M, Ypred), axis=0)
        df_c1 = rank_x - rank_x0
        df_res = n - rank_x
        ## Broadcast over ss_errors of Y
        f_stats = (SS * df_res) / (ss_errors * df_c1)
        if not pval:
            return (f_stats, None)
        else:
            p_vals = stats.f.sf(f_stats, df_c1, df_res)
            return f_stats, p_vals

class MURidgeLM:
    """Mass-univariate linear modeling based on Ridge regression.
    Fit independant Ridge models for each columns of Y."""
    
    def __init__(self, **kwargs):
        self.coef_ = None

    def fit(self, X, Y):
        pass

    def predict(self, X):
        X = safe_asarray(X)
        return np.dot(X, self.coef_)

#class OLSRegression:
#    def __init__(self):
#        pass
#
#    def transform(self, Y, y_group_indices, X, x_group_indices):
#        """
#
#        Example
#        -------
#        import numpy as np
#        import random
#        from mulm.models import OLSRegression
#        n_samples = 10
#        n_xfeatures = 20
#        n_yfeatures = 15
#        x_n_groups = 3
#        y_n_groups = 2
#
#        X = np.random.randn(n_samples, n_xfeatures)
#        Y = np.random.randn(n_samples, n_yfeatures)
#        x_group_indices = np.array([randint(0, x_n_groups)\
#            for i in xrange(n_xfeatures)])
#        y_group_indices = np.array([randint(0, y_n_groups)\
#            for i in xrange(n_yfeatures)])
#
#        regression = OLSRegression()
#        print regression.transform(Y, y_group_indices, X, x_group_indices)
#
#        """
#        ret = {}
#        y_uni_group_indices = set(y_group_indices)
#        x_uni_group_indices = set(x_group_indices)
#        for y_uni_group_index in y_uni_group_indices:
#            for x_uni_group_index in x_uni_group_indices:
#                if y_uni_group_index == -1 or x_uni_group_index == -1:
#                    continue
#                # y_uni_group_index = y_uni_group_indices.pop()
#                # x_uni_group_index = x_uni_group_indices.pop()
#                gY = Y[:, y_group_indices == y_uni_group_index]
#                gX = X[:, x_group_indices == x_uni_group_index]
#                betas, ss_errors = ols(gX, gY)
#                contrast = np.zeros(gX.shape[1])
#                contrast[0] = 1
#                tp, t = ols_stats_tcon(gX, betas, ss_errors,\
#                    contrast=contrast, pval=True)
#                fp, f = ols_stats_fcon(gX, betas, ss_errors,\
#                    contrast=contrast, pval=True)
#                key = "gy_%s_gx_%s" % (y_uni_group_index, x_uni_group_index)
#                key_t = key + "_t"
#                key_tp = key + "_tp"
#                key_f = key + "_f"
#                key_fp = key + "_fp"
#                ret[key_t] = t
#                ret[key_tp] = tp
#                ret[key_f] = f
#                ret[key_fp] = fp
#        return ret
