# -*- coding: utf-8 -*-
##########################################################################
# Created on Created on Thu Feb  6 15:15:14 2020
# Copyright (c) 2013-2021, CEA/DRF/Joliot/NeuroSpin. All rights reserved.
# @author:  Edouard Duchesnay
# @email:   edouard.duchesnay@cea.fr
# @license: BSD 3-clause.
##########################################################################

"""
Module that contains utility functions.
"""

import numpy as np
import scipy.stats


def ttest_pval(df, tstat, two_tailed=True):
    """Calculate p-values.


    Parameters
    ----------
    df : float
        Degrees of freedom.
    tstat : array (p, ), optional
        T-statistics. The default is None.
    two_tailed : bool, optional
        two-tailed (two-sided) test. The default is True.

    Returns
    -------
    pval : array (p, )
        P-values.

    Example
    -------
    >>> import numpy as np
    >>> from mulm import ttest_pval
    >>> x = [.1, .2, .3, -.1, .1, .2, .3]
    >>> pval = ttest_pval(df=len(x)-1, tstat=2.9755097944025275)
    >>> np.allclose(pval, 0.02478)
    True
    """
    if two_tailed:
        pval = 2 * scipy.stats.t.sf(np.abs(tstat), df=df)
    else:
        pval = scipy.stats.t.sf(tstat, df=df)

    return pval


def ttest_ci(df, estimate=None, tstat=None, se=None, mu=0, alpha=0.05,
             two_tailed=True):
    """Calculate confidence interval given at least two of the values within estimate, tstat, se.

    See confidence intervals: https://en.wikipedia.org/wiki/Confidence_interval

    Parameters
    ----------
    df : float
        Degrees of freedom.
    estimate : array (p, ), optional
        Estimate of the parameter. The default is None.
    tstat : array (p, ), optional
        T-statistics. The default is None.
    se : array (p, ), optional
        Standard error. The default is None.
    mu : float, optional
        Null hypothesis. The default is 0.
    alpha : float, optional
        1 - confidence level. The default is 0.05.
    two_tailed : bool, optional
        two-tailed (two-sided) test. The default is True.

    Returns
    -------
    estimate : array (p, )
        Estimates.
    tstat : array (p, )
        T-statistics.
    se : array (p, )
        Standard errors.
    pval : array (p, )
        P-values.
    ci : (array (p, ) array (p, ))
        Confidence intervals.

    Example
    -------
    >>> import numpy as np
    >>> from mulm import ttest_ci
    >>> x = [.1, .2, .3, -.1, .1, .2, .3]
    >>> estimate, se, tstat, ci = ttest_ci(df=len(x)-1, estimate=np.mean(x),
    ...                                    se=np.std(x, ddof=1)/np.sqrt(len(x)))
    >>> np.allclose((estimate, tstat, ci[0], ci[1]),
    ...             (0.15714286, 2.9755, 0.02791636, 0.28636936))
    True
    """

    # tstat = (estimate - mu) / se
    assert np.sum([s is not None for s in (estimate, tstat, se)]) >= 2,\
        "Provide at least two values within estimate, tstat, se"
    if se is None:
        se = (estimate - mu) / tstat
    elif tstat is None:
        tstat = (estimate - mu) / se
    elif estimate is None:
        estimate = tstat * se  + mu

    if two_tailed:
        cint = scipy.stats.t.ppf(1 - alpha / 2, df)
        ci = estimate - cint * se, estimate + cint * se
    else:
        cint = scipy.stats.t.ppf(1 - alpha, df)
        ci = estimate - cint * se, np.inf

    return estimate, se, tstat, ci

