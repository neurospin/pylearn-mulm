#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:15:14 2020

@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import mulm


class Residualizer:
    """
    Residualization of a Y data on possibly adjusted for other variables.
    Example: Y is a (n, p) array of p-dependant variables, we want to residualize
    for "site" adjusted for "age + sex + diagnosis"

    1) `Residualizer(data=df,
                     formula_res="site",
                     formula_full=site + age + sex + diagnosis")`
    2) `get_design_mat()` will return the numpy (n, k) array design matrix.
    Row selection can be done on both Y and design_mat (Cross-val., etc.)

    3) `fit(Y, design_mat)` fits the model:
    Y = b1 site + b2 age + b3 sex + b4 diagnosis + eps
    => learn and store b1, b2, b3 and b4

    4) `transform(Y, design_mat)` Y and design_mat can contains other
    observations than the ones used in training phase.

    Return Y - b1 site

    Parameters
    ----------
    Y: array (n, p)
        dependant variables

    formula_res: str
        Residualisation formula ex: "site"

    formula_full: str
        Full model (formula) of residualisation containing other variables
        to adjust for. Ex.: "site + age + sex + diagnosis"

    design_mat: array (n, k)
        where Y.shape[0] == design_mat.shape[0] and design_mat.shape[1] is
        the same in fit and transform

    pack_data: boolean (default=False)

    Returns
    -------
    Y: array (n, p)
        Residualized dependant variables

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import scipy.stats as stats
    >>> from mulm.residualizer import Residualizer
    >>> import seaborn as sns
    >>> np.random.seed(1)
    >>>
    >>> # Dataset with site effect on age
    >>> site = np.array([-1] * 50 + [1] * 50)
    >>> age = np.random.uniform(10, 40, size=100) + 5 * site
    >>> y = -0.1 * age  + site + np.random.normal(size=100)
    >>> data = pd.DataFrame(dict(y=y, age=age, site=site.astype(object)))
    >>>
    >>> # Simple residualization on site
    >>> res_spl = Residualizer(data, formula_res="site")
    >>> yres = res_spl.fit_transform(y[:, None], res_spl.get_design_mat())
    >>>
    >>> # Site residualization adjusted for age
    >>> res_adj = Residualizer(data, formula_res="site", formula_full="age + site")
    >>> yadj = res_adj.fit_transform(y[:, None], res_adj.get_design_mat())
    >>>
    >>> # Site residualization adjusted for age provides higher correlation,
    >>> # and lower stderr than simple residualization
    >>> lm_res = stats.linregress(age, yres.ravel())
    >>> lm_adj = stats.linregress(age, yadj.ravel())
    >>>
    >>> np.allclose((lm_res.slope, lm_res.rvalue, lm_res.stderr),
    >>>             (-0.079187578, -0.623733003, 0.0100242219))
    True
    >>> np.allclose((lm_adj.slope, lm_adj.rvalue, lm_adj.stderr),
    >>>             (-0.110779913, -0.7909219758, 0.00865778640))
    True
    """

    def __init__(self, data, formula_res, formula_full=None):

        if formula_full is None:
            formula_full = formula_res
        res_terms = mulm.design_matrix(formula=formula_res, data=data)[1].keys()
        self.design_mat, self.t_contrasts, self.f_contrasts = \
            mulm.design_matrix(formula=formula_full, data=data)
        # mask of terms in residualize formula within full model
        self.mask = np.array([cont for term, cont in self.t_contrasts.items()
                              if term in res_terms]).sum(axis=0) == 1

    def get_design_mat(self):
        return self.design_mat

    def fit(self, Y, design_mat):
        """
        Y: array (n, p)
            Dependant variables

        design_mat: array(n, k)
            Design matrix of independant variables
        """

        assert Y.shape[0] == design_mat.shape[0]
        assert self.mask.shape[0] == design_mat.shape[1]
        self.mod_mulm = mulm.MUOLS(Y, design_mat).fit()
        return self

    def transform(self, Y, design_mat=None):

        assert Y.shape[0] == design_mat.shape[0]
        assert self.mask.shape[0] == design_mat.shape[1]
        return Y - np.dot(design_mat[:, self.mask],
                          self.mod_mulm.coef[self.mask, :])

    def fit_transform(self, Y, design_mat):
        self.fit(Y, design_mat)
        return self.transform(Y, design_mat)


class ResidualizerEstimator:
    """Wrap Residualizer into an Estimator compatible with sklearn API.

    Parameters
    ----------
    residualizer: Residualizer
    """

    def __init__(self, residualizer):

        self.residualizer = residualizer
        self.design_mat_ncol = self.residualizer.design_mat.shape[1]

    def fit(self, X, y=None):
        design_mat, Y = self.upack(X)
        return self.residualizer.fit(Y, design_mat)

    def transform(self, X):
        design_mat, Y = self.upack(X)
        return self.residualizer.transform(Y, design_mat)

    def fit_transform(self, X, y=None):
        design_mat, Y = self.upack(X)
        self.residualizer.fit(Y, design_mat)
        return self.residualizer.transform(Y, design_mat)

    def pack(self, Z, X):
        """Pack (concat) Z (design matrix) and X to match scikit-learn pipelines.

        Parameters
        ----------
        Z: array (n, k)
            the design_matrix
        X: array (n, p)
            the input data for scikit-learn: fit(X, y) or transform(X)

        Returns
        -------
        (n, (k+p)) array: [design_matrix, X]
        """
        return np.hstack([Z, X])

    def upack(self, X):
        """Unpack X and Z (design matrix) from X.

        Parameters
        ----------
        X: array (n, (k+p))
            array: [design_matrix, X]

        Returns
        -------
            design_matrix, X
        """
        return X[:, :self.design_mat_ncol], X[:, self.design_mat_ncol:]


def residualize(Y, data, formula_res, formula_full=None):
    """Helper function. See Residualizer
    """
    res = Residualizer(data=data, formula_res=formula_res, formula_full=formula_full)
    return res.fit_transform(Y, res.get_design_mat())
