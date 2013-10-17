# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:06:51 2013

@author: jinpeng.li@cea.fr
"""


import numpy as np
import random
from epac import LocalEngine, SomaWorkflowEngine
from epac import ColumnSplitter
from mulm import MUOLS, MUOLSStatsCoefficients, MUOLSStatsPredictions

if __name__ == "__main__":
    # =============================================================================
    # Generate data
    n_samples = 10
    n_xfeatures = 20
    n_yfeatures = 15
    x_n_groups = 3
    y_n_groups = 2
    X = np.random.randn(n_samples, n_xfeatures)
    Y = np.random.randn(n_samples, n_yfeatures)
    x_group_indices = [1, 1, 3, 2, 2, 2, 2]
    x_group_indices = np.array([random.randint(0, x_n_groups)\
        for i in xrange(n_xfeatures)])
    blocks_size = [np.sum(lev == x_group_indices) for lev in set(x_group_indices)]
    print "Blocks size", blocks_size

    # =============================================================================
    # 1) Prediction for each X block return a n_samples x n_yfeatures
    mulm = ColumnSplitter(MUOLS(), {"X": x_group_indices})  # EPAC workflow
    mulm.run(X=X, Y=Y)                                      # map
    res_tab = mulm.reduce()                                 # reduce
    for res_tab_key in res_tab.keys():
        print "===============key=%s=============" % res_tab_key
        print "predictions shape == (n_samples, n_yfeatures) ?:",\
            res_tab[res_tab_key]['Y/pred'].shape == (n_samples, n_yfeatures)

    # =============================================================================
    # 2) coeficient Statistics for each block (of size k) return
    # a block_size x n_yfeatures array of t-values and p-values
    mulm_stats_coefficients = ColumnSplitter(MUOLSStatsCoefficients(),
                                {"X": x_group_indices})  # EPAC workflow
    mulm_stats_coefficients.run(X=X, Y=Y)                # map
    res_tab = mulm_stats_coefficients.reduce()           # reduce
    k = 0
    for res_tab_key in res_tab.keys():
        print "===============key=%s=============" % res_tab_key
        print "p-values shape: == (k, n_yfeatures) ?:",\
            res_tab[res_tab_key]['pvals'].shape == (blocks_size[k], n_yfeatures)
        k += 1

    # =============================================================================
    # 3) Prediction statistics for each block (of size k) return
    # a block_size x n_yfeatures array of t-values and p-values
    mulm_stats_predictions = ColumnSplitter(MUOLSStatsPredictions(),
                                {"X": x_group_indices})  # EPAC workflow
    mulm_stats_predictions.run(X=X, Y=Y)                 # map
    res_tab = mulm_stats_predictions.reduce()            # reduce
    for res_tab_key in res_tab.keys():
        print "===============key=%s=============" % res_tab_key
        print "p-values shape: == n_yfeatures ?:",\
            res_tab[res_tab_key]['r2'].shape == (n_yfeatures, )
