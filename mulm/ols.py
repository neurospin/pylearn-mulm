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


def ols_stats_tcon(X, betas, ss_errors, contrast, pval=False):
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
    c = np.asarray(contrast)
    n = X.shape[0]
    # t = c'beta / std(c'beta)
    # std(c'beta) = sqrt(var_err (c'X+)(X+'c))
    Xpinv = scipy.linalg.pinv(X)
    cXpinv = np.dot(c, Xpinv)
    R = np.eye(n) - np.dot(X, Xpinv)
    df = np.trace(R)
    ## Broadcast over ss errors
    var_errors = ss_errors / df
    std_cbeta = np.sqrt(var_errors * np.dot(cXpinv, cXpinv.T))
    t_stats = np.dot(c, betas) / std_cbeta
    if not pval:
        return (t_stats, None)
    else:
        p_vals = stats.t.sf(t_stats, df)
        return t_stats, p_vals


def ols_stats_fcon(X, betas, ss_errors, contrast, pval=False):
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