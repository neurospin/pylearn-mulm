# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 19:07:13 CET 2019

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import numpy as np
import pandas as pd
import patsy
from collections import OrderedDict

def design_matrix(formula, data):
    """
    Build the design matrix t-contrasts and F-contrasts given the formula and the dataframe.
    Remark: use `patsy.dmatrix(formula, data=data)`

    Parameters
    ----------

    formula, str
    data, DataFrame

    return
    ------
    X: (array) the design matrix
    t_contrasts: (OrderedDict key=colname, val=array (P,)). Remark: len(t_contrasts) == X.shape[1]
    f_contrasts: (OrderedDict key=term name in formula, val=array (P, P)),
    """

    # Build design matrix
    dmat = patsy.dmatrix(formula, data=data)

    # Build T-contrasts
    t_contrasts = OrderedDict()

    for idx, name in enumerate(dmat.design_info.column_names):
        cont = np.zeros(dmat.shape[1])
        cont[idx] = 1
        t_contrasts[name] = cont

    # Build F-contrasts
    f_contrasts = OrderedDict()

    for term, slice in dmat.design_info.term_name_slices.items():
        # if term == "site":
        #    break
        cont = np.zeros((dmat.shape[1], dmat.shape[1]))
        indices = np.arange(slice.start, slice.stop)
        cont[indices, indices] = 1
        f_contrasts[term] = cont

    return np.asarray(dmat), t_contrasts, f_contrasts
