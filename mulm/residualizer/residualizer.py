# -*- coding: utf-8 -*-
##########################################################################
# Created on Created on Thu Feb  6 15:15:14 2020
# Copyright (c) 2013-2021, CEA/DRF/Joliot/NeuroSpin. All rights reserved.
# @author:  Edouard Duchesnay
# @email:   edouard.duchesnay@cea.fr
# @license: BSD 3-clause.
##########################################################################

"""
Residualization of a Y data on possibly adjusted for other variables.
"""

import numpy as np
import mulm
import pandas as pd

class Residualizer:
    """ Residualization of a Y data on possibly adjusted for other variables.

    Example: Y is a (n, p) array of p-dependant variables, we want to residualize
    for "site" adjusted for "age + sex".

    1) Use of DataFrame and formula:
    1.1) `Residualizer(data=df, formula_res="site", formula_full=site + age + sex")`

    1.2) `Z = get_design_mat(data)` will return the numpy (n, k) array design matrix.
    Row selection can be done on both Y and design_mat (Cross-val., etc.)

    2) Use of raw arrays: if you choose to manually write your design matrix.
    In this case provide res_mask ie, the residualization mask within your full.
    model. For example: `Residualizer(mask=[False, True, False, False])` will
    fit the whole model and residualize on the second regressor, ie, site.

    3) `fit(Y, X)` fits the model:
    Y = b0 + b1 site + b2 age + b3 sex + eps
    => learn and store b1, b2, b3

    4) `transform(Y, X)` residualize Y on X, ie, returns Y - b1 site

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
    >>> yres = res_spl.fit_transform(y[:, None], res_spl.get_design_mat(data))
    >>>
    >>> # Site residualization adjusted for age
    >>> res_adj = Residualizer(data, formula_res="site", formula_full="age + site")
    >>> yadj = res_adj.fit_transform(y[:, None], res_adj.get_design_mat(data))
    >>>
    >>> # Site residualization adjusted for age provides higher correlation,
    >>> # and lower stderr than simple residualization
    >>> lm_res = stats.linregress(age, yres.ravel())
    >>> lm_adj = stats.linregress(age, yadj.ravel())
    >>>
    >>> np.allclose((lm_res.slope, lm_res.rvalue, lm_res.stderr),
    ...             (-0.079187578, -0.623733003, 0.0100242219))
    True
    >>> np.allclose((lm_adj.slope, lm_adj.rvalue, lm_adj.stderr),
    ...             (-0.110779913, -0.7909219758, 0.00865778640))
    True
    """

    def __init__(self, data=None, formula_res=None, formula_full=None,
                 contrast_res=None):
        """
        Parameters
        ----------
        data: DataFrame
            DataFrame containing column to build the design matrix (default None).

        formula_res: str
            Residualisation formula. Ex: "site" (default None).

        formula_full: str
            Full model (formula) of residualisation containing other variables
            to adjust for. Ex.: "site + age + sex" (default None).

        cont_res: boolean array
            the contrast for residualisation (matches formula_res).
            Ex: [False, True, False, False]. The default None corresponds to True
            everywhere.
        """

        if isinstance(data, pd.DataFrame) and isinstance(formula_res, str):
            if formula_full is None:
                formula_full = formula_res
            self.formula_full = formula_full
            res_terms = mulm.design_matrix(formula=formula_res, data=data)[1].keys()
            _, self.t_contrasts, self.f_contrasts = \
                mulm.design_matrix(formula=formula_full, data=data)
            # mask of terms in residualize formula within full model
            self.contrast_res = np.array([cont for term, cont in self.t_contrasts.items()
                                  if term in res_terms]).sum(axis=0) == 1
        else:
            self.contrast_res = contrast_res

    def get_design_mat(self, data):
        design_mat, t_contrasts, f_contrasts = \
            mulm.design_matrix(formula=self.formula_full, data=data)
        assert np.all([self.t_contrasts[k] == t_contrasts[k]
                       for k in self.t_contrasts]), "new data doesn't"
        return design_mat

    def fit(self, Y, X):
        """Fit parameters of p linear models where each Y is regressed on X.

        Parameters
        ----------
        Y: array (n, p)
            Dependant variables

        X: array(n, k)
            Design matrix of independant variables
        """
        if self.contrast_res is None:
            self.contrast_res = np.ones(X.shape[1]).astype(bool)

        assert Y.shape[0] == X.shape[0]
        assert self.contrast_res.shape[0] == X.shape[1], "contrast doesn't match design matrix"
        self.mod_mulm = mulm.MUOLS(Y, X).fit()
        return self

    def transform(self, Y, X):
        """Residualize Y on X.

        Parameters
        ----------
        Y: array (n, p)
            Dependant variables

        X: array(n, k)
            Design matrix of independant variables

        Returns
        -------
        Yres: array (n, p)
            Residualized Y data.
        """
        assert Y.shape[0] == X.shape[0]
        assert self.contrast_res.shape[0] == X.shape[1], "contrast doesn't match design matrix"
        return Y - np.dot(X[:, self.contrast_res],
                          self.mod_mulm.coef[self.contrast_res, :])

    def fit_transform(self, Y, X):
        """Fit parameters of p linear models where each Y is regressed on X.
        Residualize Y on X.

        Parameters
        ----------
        Y: array (n, p)
            Dependant variables

        X: array(n, k)
            Design matrix of independant variables

        Returns
        -------
        Yres: array (n, p)
            Residualized Y data.
        """
        self.fit(Y, X)
        return self.transform(Y, X)


def residualize(Y, data, formula_res, formula_full=None):
    """Helper function. See Residualizer.
    """
    res = Residualizer(data=data, formula_res=formula_res, formula_full=formula_full)
    return res.fit_transform(Y, res.get_design_mat(data))


class ResidualizerEstimator:
    """Wrap Residualizer into an estimator compatible with sklearn API.

    Note that to be consistant with sklearn API, here X contains the input variable
    and Z is the design matrix for residualization.
    """

    def __init__(self, residualizer):
        """
        Parameters
        ----------
        residualizer: Residualizer
        """
        self.residualizer = residualizer
        self.design_mat_ncol = self.residualizer.contrast_res.shape[0]

    def fit(self, X, y=None):
        Z, Y = self.upack(X)
        return self.residualizer.fit(Y, Z)

    def transform(self, X):
        Z, Y = self.upack(X)
        return self.residualizer.transform(Y, Z)

    def fit_transform(self, X, y=None):
        Z, Y = self.upack(X)
        self.residualizer.fit(Y, Z)
        return self.residualizer.transform(Y, Z)

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

    def upack(self, ZX):
        """Unpack X and Z (design matrix) from X.

        Parameters
        ----------
        ZX: array (n, (k+p))
            array: [Z, X]

        Returns
        -------
            Z (design_matrix), X
        """
        return ZX[:, :self.design_mat_ncol], ZX[:, self.design_mat_ncol:]

