# -*- coding: utf-8 -*-
##########################################################################
# Created on Created on Thu Feb  6 15:15:14 2020
# Copyright (c) 2013-2021, CEA/DRF/Joliot/NeuroSpin. All rights reserved.
# @author:  Edouard Duchesnay
# @email:   edouard.duchesnay@cea.fr
# @license: BSD 3-clause.
##########################################################################

"""
Module with helpers functions.
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

    Returns
    -------
    X: array
        the design matrix
    t_contrasts: OrderedDict key=colname, val=array (P,)
        remark: len(t_contrasts) == X.shape[1]
    f_contrasts: OrderedDict key=term name in formula, val=array (P, P)
        not documented.
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
