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


from epac.workflow.splitters import BaseNodeSplitter
from epac.workflow.splitters import Slicer
from epac.workflow.splitters import VirtualList
from epac.workflow.factory import NodeFactory
from epac.configuration import conf


class MULMColSlicer(Slicer):
    """Row-wise reslicing of the downstream blocs.

    Parameters
    ----------
    name: string

    apply_on: string or list of strings (or None)
        The name(s) of the downstream blocs to be rescliced. If
        None, all downstream blocs are rescliced.
    """

    def __init__(self, signature_name, nb, apply_on):
        super(self.__class__, self).__init__(signature_name, nb)
        self.slices = None
        self.n = 0  # the dimension of that array in ds should respect
        if not apply_on:  # None is an acceptable value here
            self.apply_on = apply_on
        elif isinstance(apply_on, list):
            self.apply_on = apply_on
        elif isinstance(apply_on, str):
            self.apply_on = [apply_on]
        else:
            raise ValueError("apply_on must be a string or a "\
                "list of strings or None")

    def set_sclices(self, slices):
        # convert as a list if required
        if isinstance(slices, dict):
            self.slices =\
                {k: slices[k].tolist() if isinstance(slices[k], np.ndarray)
                else slices[k] for k in slices}
            self.n = np.sum([len(v) for v in self.slices.values()])
        else:
            self.slices = \
                slices.tolist() if isinstance(slices, np.ndarray) else slices
            self.n = len(self.slices)

    def transform(self, **Xy):
        if not self.slices:
            raise ValueError("Slicing hasn't been initialized. "
            "Slicers constructors such as CV or Perm should be called "
            "with a sample. Ex.: CV(..., y=y), Perm(..., y=y)")
        data_keys = self.apply_on if self.apply_on else Xy.keys()
        for data_key in data_keys:  # slice input data
            dat = Xy.pop(data_key)
            if len(dat.shape) == 2:
                Xy[data_key] = dat[:, self.slices[data_key]]
            else:
                Xy[data_key] = dat[self.slices[data_key]]
        return Xy


class MULM(BaseNodeSplitter):
    """Cross-validation parallelization.

    Parameters
    ----------
    node: Node | Estimator
        Estimator: should implement fit/predict/score function
        Node: Pipe | Par*

    n_folds: int
        Number of folds. (Default 5)

    cv_type: string
        Values: "stratified", "random", "loo". Default "stratified".

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    reducer: Reducer
        A Reducer should implement the reduce(node, key2, val) method.
        Default ClassificationReport() with default arguments.
    
    Example
    -------
    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> from epac import Methods
    >>> from epac.workflow.splitters import CVBestSearchRefitParallel

    """

    def __init__(self, node, x_group_indices, y_group_indices, **kwargs):
        super(self.__class__, self).__init__()

        self.x_group_indices = x_group_indices
        self.y_group_indices = y_group_indices
        self.slicer = MULMColSlicer(
            signature_name=self.__class__.__name__,\
            nb=0,\
            apply_on=None)

        self.x_uni_group_indices = set(x_group_indices)
        self.y_uni_group_indices = set(y_group_indices)
        size = len(self.x_uni_group_indices) * len(self.y_uni_group_indices)
        self.children = VirtualList(size=size, parent=self)
        self.slicer.parent = self

        subtree = NodeFactory.build(node)
        # subtree = node if isinstance(node, BaseNode) else LeafEstimator(node)
        self.slicer.add_child(subtree)

    def move_to_child(self, nb):
        self.slicer.set_nb(nb)
        cpt = 0
        cp_x_uni_group_indice = None
        cp_y_uni_group_indice = None
        for x_uni_group_indice in self.x_uni_group_indices:
            if (not cp_x_uni_group_indice == None)\
                or (not cp_y_uni_group_indice == None):
                break
            for y_uni_group_indice in self.y_uni_group_indices:
                if cpt == nb:
                    cp_x_uni_group_indice = x_uni_group_indice
                    cp_y_uni_group_indice = y_uni_group_indice
                    break
                cpt += 1
        if (not cp_x_uni_group_indice == None)\
                or (not cp_y_uni_group_indice == None):
            x_indices = np.nonzero(np.asarray(self.x_group_indices) == \
                            np.asarray(cp_x_uni_group_indice))
            x_indices = x_indices[0]
            y_indices = np.nonzero(np.asarray(self.y_group_indices) == \
                            np.asarray(cp_y_uni_group_indice))
            y_indices = y_indices[0]
            self.slicer.set_sclices({"X": x_indices,
                                     "Y": y_indices})
        return self.slicer

    def transform(self, **Xy):
        self._sclices = None
        return Xy

    def get_parameters(self):
        return dict(n_folds=self.n_folds)


if __name__ == "__main__":
    import numpy as np
    import random
    from mulm.models import MULM
    from sklearn import datasets
    from sklearn.svm import SVC
    from epac import Methods
    from epac.workflow.splitters import CVBestSearchRefitParallel

    n_samples = 10
    n_xfeatures = 20
    n_yfeatures = 15
    x_n_groups = 3
    y_n_groups = 2

    X = np.random.randn(n_samples, n_xfeatures)
    Y = np.random.randn(n_samples, n_yfeatures)
    x_group_indices = np.array([random.randint(0, x_n_groups)\
        for i in xrange(n_xfeatures)])
    y_group_indices = np.array([random.randint(0, y_n_groups)\
        for i in xrange(n_yfeatures)])

    mulm = MULM(MUOLS(), x_group_indices, y_group_indices)
    ret_XY = mulm.run(X=X, Y=Y)
    print "==================ret_XY======================"
    print ret_XY
    for leaf in mulm.walk_leaves():
        print "===============leaf.load_results()================="
        print "key =", leaf.get_key()
        print leaf.load_results()

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
