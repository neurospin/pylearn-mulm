# -*- coding: utf-8 -*-
##########################################################################
# Created on Tue Jun 25 13:25:41 2013
# Copyright (c) 2013-2021, CEA/DRF/Joliot/NeuroSpin. All rights reserved.
# @author:  Edouard Duchesnay
# @email:   edouard.duchesnay@cea.fr
# @license: BSD 3-clause.
##########################################################################

"""
Linear models for massive univariate statistics.
"""

import numpy as np
import scipy
from sklearn.preprocessing import scale
from scipy import stats
from .utils import ttest_pval
#from mulm.utils import estimate_se_tstat_pval_ci

class MUPairwiseCorr:
    """Mass-univariate pairwise correlations. Given two arrays X (n_samples x p)
    and Y (n_samples x q). Fit p x q independent linear models. Prediction
    and stats return (p x q) array.

    Examples
    --------
    >>> import numpy as np
    >>> from mulm import MUPairwiseCorr
    >>> X = np.random.randn(10, 5)
    >>> Y = np.random.randn(10, 3)
    >>> corr = MUPairwiseCorr(X, Y).fit()
    >>> f, p, df = corr.stats_f()
    >>> print(f.shape)
    (5, 3)
    """
    def __init__(self, X, Y):
        """
        Parameters
        ----------
        Y : numpy array (n_samples, p)
            First block of variables.

        X : numpy array (n_samples, q)
            Second block of variables.
        """
        self.Xs = scale(X, copy=True) # TODO PERFORM BASIC CHECK ARRAY
        self.Ys = scale(Y, copy=True) # TODO PERFORM BASIC CHECK ARRAY
        self.n_samples = X.shape[0]

        if X.shape[0] != Y.shape[0]:
            raise ValueError('matrices are not aligned')

    def fit(self):
        self.Corr_ = np.dot(self.Xs.T, self.Ys) / self.n_samples
        return self

    def predict(self, X):
        pass

    def stats_f(self, pval=True):
        """

        Parameters
        ----------
        pval

        Returns
        -------
        fstats (k, p) array, pvals (k, p) array, df (k,) array
        """
        R2 = self.Corr_ ** 2
        df_res = self.n_samples - 2
        f_stats = R2 * df_res / (1 - R2)
        if not pval:
            return (f_stats, None)
        else:
            p_vals = stats.f.sf(f_stats, 1, df_res)
            return f_stats, p_vals, df_res


class MUOLS:
    """Mass-univariate linear modeling based Ordinary Least Squares.
    Given two arrays X (n_samples, p) and Y (n_samples, q).
    Fit q independent linear models, ie., for all y in Y fit: lm(y ~ X).

    Example
    -------
    >>> import numpy as np
    >>> import mulm
    >>> np.random.seed(42)
    >>> # n_samples, nb of features that depends on X and that are pure noise
    >>> n_samples, n_info, n_noise = 100, 2, 100
    >>> beta = np.array([1, 0, 0.5, 0, 2])[:, np.newaxis]
    >>> X = np.random.randn(n_samples, 5) # Design matrix
    >>> X[:, -1] = 1 # Intercept
    >>> Y = np.random.randn(n_samples, n_info + n_noise)
    >>> Y[:, :n_info] += np.dot(X, beta) # n_info features depend from X
    >>> contrasts = np.identity(X.shape[1])[:4] # don't test the intercept
    >>> mod = mulm.MUOLS(Y, X).fit()
    >>> tvals, pvals, df = mod.t_test(contrasts, two_tailed=True)
    >>> print(pvals.shape)
    (4, 102)
    >>> print("Nb of uncorrected p-values <5%:", np.sum(pvals < 0.05))
    Nb of uncorrected p-values <5%: 18
    """

    def __init__(self, Y, X):
        """
        Parameters
        ----------
        Y : numpy array (n_samples, p)
            dependant (target) variables.

        X : numpy array (n_samples, q)
            design matrix.
        """
        self.coef = None
        if X.shape[0] != Y.shape[0]:
            raise ValueError('matrices are not aligned')
        self.X = X  # TODO PERFORM BASIC CHECK ARRAY
        self.Y = Y  # TODO PERFORM BASIC CHECK ARRAY

    def _block_slices(self, dim_size, block_size):
        """Generator that yields slice objects for indexing into
        sequential blocks of an array along a particular axis
        """
        count = 0
        while True:
            yield slice(count, count + block_size, 1)
            count += block_size
            if count >= dim_size:
                return

    def fit(self, block=False, max_elements=2 ** 27):
        """Fit p independent linear models, ie., for all y in Y fit: lm(y ~ X).

        Parameters
        ----------
        block : boolean
            Use block=True for huge matrices Y.
            Operations block by block to optimize time and memory.

        max_elements : int
            block dimension (2**27 corresponds to 1Go)

        Returns
        -------
            self
        """
        self.block = block
        self.max_elements = max_elements
        self.pinv = scipy.linalg.pinv(self.X)
        n, p = self.Y.shape
        q = self.X.shape[1]
        if self.block:
            if self.max_elements < n:
                raise ValueError('the maximum number of elements is too small')
            max_cols = int(self.max_elements / n)
        else:
            max_cols = p
        self.coef = np.zeros((q, p))
        self.err_ss = np.zeros(p)
        for pp in self._block_slices(p, max_cols):
            if isinstance(self.Y, np.memmap):
                Y_block = self.Y[:, pp].copy()  # copy to force a read
            else: Y_block = self.Y[:, pp]
            #Y_block = self.Y[:, pp]
            self.coef[:, pp] = np.dot(self.pinv, Y_block)
            y_hat = np.dot(self.X, self.coef[:, pp])
            err = Y_block - y_hat
            del Y_block, y_hat
            self.err_ss[pp] = np.sum(err ** 2, axis=0)
            del err

        return self

    def predict(self, X):
        """Predict Y given a new design matrix X.

        Parameters
        ----------
        X : numpy array (n_samples, q)
            design matrix of new predictors.

        Returns
        -------
        (n_samples, 1) array of predicted values (X beta)
        """
        #from sklearn.utils import safe_asarray
        import numpy as np
        #X = safe_asarray(X) # TODO PERFORM BASIC CHECK ARRAY
        pred_y = np.dot(X, self.coef)
        return pred_y


    def t_test(self, contrasts, pval=False, two_tailed=True):
        """Compute T-statistics (t-scores and p-value associated to contrast).
           The code has been cloned from the SPM MATLAB implementation.

        Parameters
        ----------
        contrasts: array (q, ) or list of arrays or array 2D.
            Single contrast (array) or list of contrasts or array of contrasts.
            The k contrasts to be tested.

        pval: boolean
            compute pvalues (default is false)

        two_tailed: boolean
            one-tailed test or a two-tailed test (default True)

        Returns
        -------
        tstats (k, p) array, pvals (k, p) array, df (k,) array

        Example
        -------
        >>> import numpy as np
        >>> import mulm
        >>> np.random.seed(42)
        >>> # n_samples, nb of features that depends on X and that are pure noise
        >>> n_samples, n_info, n_noise = 100, 2, 100
        >>> beta = np.array([1, 0, 0.5, 0, 2])[:, np.newaxis]
        >>> X = np.random.randn(n_samples, 5) # Design matrix
        >>> X[:, -1] = 1 # Intercept
        >>> Y = np.random.randn(n_samples, n_info + n_noise)
        >>> Y[:, :n_info] += np.dot(X, beta) # n_info features depend from X
        >>> contrasts = np.identity(X.shape[1])[:4] # don't test the intercept
        >>> mod = mulm.MUOLS(Y, X).fit()
        >>> tvals, pvals, df = mod.t_test(contrasts, two_tailed=True)
        >>> print(pvals.shape)
        (4, 102)
        >>> print("Nb of uncorrected p-values <5%:", np.sum(pvals < 0.05))
        Nb of uncorrected p-values <5%: 18
        """
        contrasts = np.atleast_2d(np.asarray(contrasts))
        n = self.X.shape[0]
        t_stats_ = list()
        p_vals_ = list()
        df_ = list()
        for contrast in contrasts:
            # contrast = contrasts[0]
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
                p_vals = ttest_pval(df=df, tstat=t_stats, two_tailed=two_tailed)
                # #
                # if two_tailed:
                #     p_vals = stats.t.sf(np.abs(t_stats), df) * 2
                # else:
                #     p_vals = stats.t.sf(t_stats, df)
            t_stats_.append(t_stats)
            p_vals_.append(p_vals)
            df_.append(df)
        return np.asarray(t_stats_), np.asarray(p_vals_), np.asarray(df_)

    def t_test_maxT(self, contrasts, nperms=1000, two_tailed=True, **kwargs):
        """Correct for multiple comparisons using Westfall and Young, 1993 a.k.a maxT procedure.
           It is based on permutation tests procedure. This is the procedure used by FSL (https://fsl.fmrib.ox.ac.uk/).

           It should be used when the test statistics, and hence the unadjusted p-values, are dependent.
           This is the case when groups of dependant variables (in Y) tend to have highly correlated measures.
           Westfall and Young (1993) proposed adjusted p-values for less conservative multiple testing procedures which
           take into account the dependence structure among test statistics.
           References:
           - Anderson M. Winkler "Statistical analysis of areal quantities in the brain through
           permutation tests" Ph.D 2017.
           - Dudoit et al. "Multiple Hypothesis Testing in Microarray Experiments", Statist. Sci. 2003

        Parameters
        ----------
        contrasts: array (q, ) or list of arrays or array 2D.
            Single contrast (array) or list of contrasts or array of contrasts.
            The k contrasts to be tested.

        nperms: int
                permutation tests (default 1000).

        two_tailed: boolean
            one-tailed test or a two-tailed test (default True)

        Returns
        -------
        tstats (k, p) array, pvals (k, p) array corrected for multiple comparisons
        df (k,) array.

        Examples
        --------
        >>> import numpy as np
        >>> import mulm
        >>> np.random.seed(42)
        >>> # n_samples, nb of features that depends on X and that are pure noise
        >>> n_samples, n_info, n_noise = 100, 2, 100
        >>> beta = np.array([1, 0, 0.5, 0, 2])[:, np.newaxis]
        >>> X = np.random.randn(n_samples, 5) # Design matrix
        >>> X[:, -1] = 1 # Intercept
        >>> Y = np.random.randn(n_samples, n_info + n_noise)
        >>> Y[:, :n_info] += np.dot(X, beta) # n_info features depend from X
        >>> contrasts = np.identity(X.shape[1])[:4] # don't test the intercept
        >>> mod = mulm.MUOLS(Y, X).fit()
        >>> tvals, pvals, df = mod.t_test(contrasts, two_tailed=True)
        >>> print(pvals.shape)
        (4, 102)
        >>> print("Nb of uncorrected p-values <5%:", np.sum(pvals < 0.05))
        Nb of uncorrected p-values <5%: 18
        >>> tvals, pvals_corrmaxT, df = mod.t_test_maxT(contrasts, two_tailed=True)
        >>> print("Nb of corrected pvalues <5%:", np.sum(pvals_corrmaxT < 0.05))
        Nb of corrected pvalues <5%: 4
       """
        #contrast = [0, 1] + [0] * (X.shape[1] - 2)
        contrasts = np.atleast_2d(np.asarray(contrasts))
        tvals, _, df = self.t_test(contrasts=contrasts, pval=False, **kwargs)
        max_t = list()
        for i in range(nperms):
            perm_idx = np.random.permutation(self.X.shape[0])
            Xp = self.X[perm_idx, :]
            muols = MUOLS(self.Y, Xp).fit(block=self.block,
                                          max_elements=self.max_elements)
            tvals_perm, _, _ = muols.t_test(contrasts=contrasts, pval=False,
                                            two_tailed=two_tailed)
            if two_tailed:
                tvals_perm = np.abs(tvals_perm)
            max_t.append(np.max(tvals_perm, axis=1))
            del muols
        max_t = np.array(max_t)
        tvals_ = np.abs(tvals) if two_tailed else tvals
        pvalues = np.array(
            [np.array([np.sum(max_t[:, con] >= t) for t in tvals_[con, :]])\
                / float(nperms) for con in range(contrasts.shape[0])])
        return tvals, pvalues, df

    def t_test_minP(self, contrasts, nperms=10000, two_tailed=True, **kwargs):
        """Correct for multiple comparisons using minP procedure.

           References:
           - Dudoit et al. "Multiple Hypothesis Testing in Microarray Experiments", Statist. Sci. 2003

        Parameters
        ----------
        contrasts: array (q, ) or list of arrays or array 2D.
            Single contrast (array) or list of contrasts or array of contrasts.
            The k contrasts to be tested.

        nperms: int
                permutation tests (default 10000).

        two_tailed: boolean
            one-tailed test or a two-tailed test (default True)

        Returns
        -------
        tstats (k, p) array, pvals (k, p) array corrected for multiple comparisons
        df (k,) array.

        Examples
        --------
        >>> import numpy as np
        >>> import mulm
        >>> np.random.seed(42)
        >>> # n_samples, nb of features that depends on X and that are pure noise
        >>> n_samples, n_info, n_noise = 100, 2, 100
        >>> beta = np.array([1, 0, 0.5, 0, 2])[:, np.newaxis]
        >>> X = np.random.randn(n_samples, 5) # Design matrix
        >>> X[:, -1] = 1 # Intercept
        >>> Y = np.random.randn(n_samples, n_info + n_noise)
        >>> Y[:, :n_info] += np.dot(X, beta) # n_info features depend from X
        >>> contrasts = np.identity(X.shape[1])[:4] # don't test the intercept
        >>> mod = mulm.MUOLS(Y, X).fit()
        >>> tvals, pvals, df = mod.t_test(contrasts, two_tailed=True)
        >>> print(pvals.shape)
        (4, 102)
        >>> print("Nb of uncorrected p-values <5%:", np.sum(pvals < 0.05))
        Nb of uncorrected p-values <5%: 18
        >>> tvals, pval_corrminp, df = mod.t_test_minP(contrasts, two_tailed=True)
        >>> print("Nb of corrected pvalues <5%:", np.sum(pval_corrminp < 0.05))
        Nb of corrected pvalues <5%: 4
        """
        tvals, pvals, df = self.t_test(contrasts=contrasts, pval=True, **kwargs)
        min_p = np.ones((contrasts.shape[0], nperms))
        perm_idx = np.zeros((self.X.shape[0], nperms + 1), dtype='int')
        for i in range(self.Y.shape[1]):
            Y_curr = self.Y[:, i]
            Yp_curr = np.zeros((self.X.shape[0], nperms + 1))

            for j in range(nperms + 1):
                if i == 0:
                    perm_idx[:, j] = np.random.permutation(self.X.shape[0])
                Yp_curr[:, j] = Y_curr[perm_idx[:, j]]
            muols = MUOLS(Yp_curr, self.X).fit()
            tvals_perm, _, _ = muols.t_test(contrasts=contrasts, pval=False,
                                            two_tailed=two_tailed)
            if two_tailed:
                tvals_perm = np.abs(tvals_perm)
            pval_perm = np.array(
               [np.array([((np.sum(tvals_perm[con, :] >= tvals_perm[con, k])) - 1) \
                         for k in range(nperms)]) / float(nperms) \
                             for con in range(contrasts.shape[0])])
            min_p = np.array(
               [(np.min(np.vstack((min_p[con, :], pval_perm[con, :])), axis=0)) \
                         for con in range(contrasts.shape[0])])
        pvalues = np.array(
               [np.array([np.sum(min_p[con, :] <= p) \
                         for p in pvals[con, :]]) / float(nperms) \
                             for con in range(contrasts.shape[0])])
        return tvals, pvalues, df

    def f_test(self, contrast, pval=False):
        """Compute F-statistics (F-scores and p-value associated to contrast).
           The code has been cloned from the SPM MATLAB implementation.

        Parameters
        ----------
        contrasts: array (q, ) or list of arrays or array 2D.
            Single contrast (array) or list of contrasts or array of contrasts.
            The k contrasts to be tested.

        pval: boolean
            compute pvalues (default is false)

        two_tailed: boolean
            one-tailed test or a two-tailed test (default True)

        Returns
        -------
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
        >>> fvals, pvals, df = mod.f_test(contrasts, pval=True)
        """
        C1 = np.atleast_2d(contrast).T
        n, p = self.X.shape
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
        y_hat = self.predict(self.X)
        SS = np.sum(y_hat * np.dot(M, y_hat), axis=0)
        df_c1 = rank_x - rank_x0
        df_res = n - rank_x
        ## Broadcast over self.err_ss of Y
        f_stats = (SS * df_res) / (self.err_ss * df_c1)
        if not pval:
            return f_stats, None, df_res
        else:
            p_vals = stats.f.sf(f_stats, df_c1, df_res)
            return f_stats, p_vals, df_res

    def stats_f_coefficients(self, X, Y, contrast, pval=False):
        return self.stats_f(contrast, pval=pval)
