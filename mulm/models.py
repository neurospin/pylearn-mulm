# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:25:41 2013

@author: ed203246
"""


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
        from sklearn.utils import safe_asarray
        import scipy
        import numpy as np
        X = safe_asarray(X)
        Y = safe_asarray(Y)
        self.coef_ = np.dot(np.linalg.pinv(X), Y)
        # self.coef_ = np.dot(scipy.linalg.pinv(X), Y)
        # self.coef_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        from sklearn.utils import safe_asarray
        import numpy as np
        X = safe_asarray(X)
        pred_y = np.dot(X, self.coef_)
        return pred_y

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
        import scipy
        import numpy as np
        from scipy import stats
        import numpy as np
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
        import numpy as np
        from sklearn.utils import array2d
        import scipy
        from scipy import stats
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
        from sklearn.utils import safe_asarray
        import numpy as np
        X = safe_asarray(X)
        return np.dot(X, self.coef_)


class MUOLSStats:
    def __init__(self):
        self.muols = MUOLS()
    def transform(self, X, Y):
        import numpy as np
        #X = np.random.randn(100, 2)
        #Y = np.hstack([np.dot(X, [1, 2])[:, np.newaxis], np.random.randn(100, 3)])
        self.muols.fit(X, Y)
        pvals = list()
        tvals = list()
        for j in xrange(X.shape[1]):
            contrast = np.zeros(X.shape[1])
            contrast[j] += 1
            t, p = self.muols.t_stats(X, Y, contrast=contrast, pval=True)
            tvals.append(t)
            pvals.append(p)
        pvals = np.asarray(pvals)
        tvals = np.asarray(tvals)
        # "transform" should return a dictionary
        return {"tvals": tvals, "pvals": pvals}


class MUOLSYR2:
    """Compute r2 (explain variance)
    See example in ./examples/permutations.py
    """
    def __init__(self):
        self.muols = MUOLS()

    def transform(self, X, Y):
        # definition of Explained Variation of R2
        # http://www.stat.columbia.edu/~gelman/research/published/rsquared.pdf
        import numpy as np
        import scipy
        self.muols.fit(X, Y)
        Ypred = self.muols.predict(X)
        var_epsilon = scipy.var(Y - Ypred, axis=0)
        var_Y = scipy.var(Y, axis=0)
        r2 = 1.0 - var_epsilon/var_Y
        return {"r2": r2}

