# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:25:41 2013

@author: ed203246
"""
import numpy as np
import scipy
from sklearn.preprocessing import scale
from scipy import stats

class MUPairwiseCorr:
    """Mass-univariate pairwise correlations. Given two arrays X [n_samples x p]
    and Y [n_samples x q]. Fit p x q independent linear models. Prediction
    and stats return [p x q] array.


    Example
    -------
    >>> import numpy as np
    >>> from mulm import MUPairwiseCorr
    >>> X = np.random.randn(10, 5)
    >>> Y = np.random.randn(10, 3)
    >>> corr = MUPairwiseCorr()
    >>> corr.fit(X, Y)
    <mulm.models.MUPairwiseCorr instance at 0x30da878>
    >>> f, p = corr.stats_f(X, Y)
    >>> print f.shape
    (5, 3)
    """
    def __init__(self, **kwargs):
        pass

    def fit(self, X, Y):
        Xs = scale(X, copy=True)
        Ys = scale(Y, copy=True)
        self.n_samples = X.shape[0]
        self.Corr_ = np.dot(Xs.T, Ys) / self.n_samples
        return self

    def predict(self, X):
        pass

    def stats_f(self, pval=True):
        R2 = self.Corr_ ** 2
        df_res = self.n_samples - 2
        f_stats = R2 * df_res / (1 - R2)
        if not pval:
            return (f_stats, None)
        else:
            p_vals = stats.f.sf(f_stats, 1, df_res)
            return f_stats, p_vals

class MUOLS:
    """Mass-univariate linear modeling based Ordinary Least Squares.
    Given two arrays X (n_samples, p) and Y (n_samples, q).
    Fit q independent linear models.

    Example
    -------
    """
    def __init__(self, Y, X):
        from sklearn.utils import safe_asarray
        self.coef = None
        self.X = safe_asarray(X)
        self.Y = safe_asarray(Y)

    def fit(self):
        self.pinv = scipy.linalg.pinv(self.X)
        self.coef = np.dot(self.pinv, self.Y)
        self.y_hat = self.predict(self.X)
        self.err = self.Y - self.y_hat
        self.err_ss = np.sum(self.err ** 2, axis=0)
        return self

    def predict(self, X):
        from sklearn.utils import safe_asarray
        import numpy as np
        X = safe_asarray(X)
        pred_y = np.dot(X, self.coef)
        return pred_y


    def t_test(self, contrasts, pval=False, two_tailed=True):
        """Compute statistics (t-scores and p-value associated to contrast)

        Parameters
        ----------
        contrasts: The k contrasts to be tested, some list or array
        that can be casted to an k x p array.

        pval: boolean
            compute pvalues (default is false)

        two_tailed: boolean
            one-tailed test or a two-tailed test (default True)

        Return
        ------
        tstats (k, p) array, pvals (k, p) array, df (k,) array

        Example
        -------
        >>> import numpy as np
        >>> import mulm
        >>> X = np.random.randn(100, 5)
        >>> Y = np.random.randn(100, 10)
        >>> beta = np.random.randn(5, 1)
        >>> Y[:, :2] += np.dot(X, beta)
        >>> contrasts = np.identity(X.shape[1])
        >>> mod = mulm.MUOLS(Y, X).fit()
        >>> tvals, pvals, df = mod.t_test(contrasts, pval=True, two_tailed=True)
        """
        contrasts = np.atleast_2d(np.asarray(contrasts))
        n = self.X.shape[0]
        t_stats_ = list()
        p_vals_ = list()
        df_ = list()
        for contrast in contrasts:
            #ccontrasts = np.asarray(contrasts)
            # t = c'beta / std(c'beta)
            # std(c'beta) = sqrt(var_err (c'X+)(X+'c))
            #Xpinv = scipy.linalg.pinv(X)
            cXpinv = np.dot(contrast, self.pinv)
            R = np.eye(n) - np.dot(self.X, self.pinv)
            df = np.trace(R)
            ## Broadcast over ss errors
            var_errors = self.err_ss / df
            std_cbeta = np.sqrt(var_errors * np.dot(cXpinv, cXpinv.T))
            t_stats = np.dot(contrast, self.coef) / std_cbeta
            p_vals = None
            if pval is not None:
                if two_tailed:
                    p_vals = stats.t.sf(np.abs(t_stats), df) * 2
                else:
                    p_vals = stats.t.sf(t_stats, df)
            t_stats_.append(t_stats)
            p_vals_.append(p_vals)
            df_.append(df)
        return np.asarray(t_stats_), np.asarray(p_vals_), np.asarray(df_)

    def t_test_maxT(self, contrasts, nperms=1000, **kwargs):
        """Correct for multiple comparisons using maxT procedure. See t_test()
        For all parameters.

        Example
        -------
        >>> import numpy as np
        >>> import mulm
        >>> import pylab as plt
        >>> n = 100
        >>> px = 5
        >>> py_info = 2
        >>> py_noize = 100
        >>> beta = np.array([1, 0, .5] + [0] * (px - 4) + [2]).reshape((px, 1))
        >>> X = np.hstack([np.random.randn(n, px-1), np.ones((n, 1))]) # X with intercept
        >>> Y = np.random.randn(n, py_info + py_noize)
        >>> Y[:, :py_info] += np.dot(X, beta)
        >>> contrasts = np.identity(X.shape[1])
        >>> mod = mulm.MUOLS(Y, X)
        >>> tvals, maxT, df = mod.t_test_maxT(contrasts, two_tailed=True)
        """
        #contrast = [0, 1] + [0] * (X.shape[1] - 2)
        tvals, _, df = self.t_test(contrasts=contrasts, pval=False, **kwargs)
        max_t = list()
        for i in xrange(nperms):
            perm_idx = np.random.permutation(self.X.shape[0])
            Xp = self.X[perm_idx, :]
            muols = MUOLS(self.Y, Xp).fit()
            tvals_perm, _, _ = muols.t_test(contrasts=contrasts, pval=False, **kwargs)
            max_t.append(np.max(tvals_perm, axis=1))
        max_t = np.array(max_t)
        pvalues = np.array(
            [np.array([np.sum(max_t[:, con] >= t) for t in tvals[con, :]])\
                / float(nperms) for con in xrange(contrasts.shape[0])])
        return tvals, pvalues, df

    def f_test(self, contrast, pval=False):
        from sklearn.utils import array2d
        #Ypred = self.predict(self.X)
        #betas = self.coef
        #ss_errors = np.sum((self.Y - self.y_hat) ** 2, axis=0)
        C1 = array2d(contrast).T
        n, p = self.X.shape
        #Xpinv = scipy.linalg.pinv(X)
        rank_x = np.linalg.matrix_rank(self.pinv)
        C0 = np.eye(p) - np.dot(C1, scipy.linalg.pinv(C1))  # Ortho. cont. to C1
        X0 = np.dot(self.X, C0)  # Design matrix of the reduced model
        X0pinv = scipy.linalg.pinv(X0)
        rank_x0 = np.linalg.matrix_rank(X0pinv)
        # Find the subspace (X1) of Xc1, which is orthogonal to X0
        # The projection matrix M due to X1 can be derived from the residual
        # forming matrix of the reduced model X0
        # R0 is the residual forming matrix of the reduced model
        R0 = np.eye(n) - np.dot(X0, X0pinv)
        # R is the residual forming matrix of the full model
        R = np.eye(n) - np.dot(self.X, self.pinv)
        # compute the projection matrix
        M = R0 - R
        #Ypred = np.dot(self.X, betas)
        SS = np.sum(self.y_hat * np.dot(M, self.y_hat), axis=0)
        df_c1 = rank_x - rank_x0
        df_res = n - rank_x
        ## Broadcast over self.err_ss of Y
        f_stats = (SS * df_res) / (self.err_ss * df_c1)
        if not pval:
            return (f_stats, None)
        else:
            p_vals = stats.f.sf(f_stats, df_c1, df_res)
            return f_stats, p_vals

    def stats_f_coefficients(self, X, Y, contrast, pval=False):
        return self.stats_f(contrast, pval=pval)

