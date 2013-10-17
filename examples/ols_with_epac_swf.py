# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:06:51 2013

@author: jinpeng.li@cea.fr
"""


import numpy as np
import random
from epac import SomaWorkflowEngine
from epac import ColumnSplitter
from mulm import MUOLS, MUOLSStatsCoefficients, MUOLSStatsPredictions
import joblib

if __name__ == "__main__":
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
    joblib.dump(x_group_indices, "/tmp/x_group_indices")

    blocks_size = [np.sum(lev == x_group_indices)
                    for lev in set(x_group_indices)]
    print "Blocks size", blocks_size

    # 1) Prediction for each X block return a n_samples x n_yfeatures
    mulm = ColumnSplitter(MUOLS(),
                          {"X": x_group_indices})
    swf_engine = SomaWorkflowEngine(tree_root=mulm,
                                    num_processes=2)
    swf_engine.export_to_gui("/tmp/mulm", X=X, Y=Y)

    # 2) coeficient Statistics for each block (of size k) return
    # a block_size x n_yfeatures array of t-values and p-values
    mulm_stats_coefficients = ColumnSplitter(MUOLSStatsCoefficients(),
                                {"X": x_group_indices})
    swf_engine = SomaWorkflowEngine(tree_root=mulm_stats_coefficients,
                                    num_processes=2)
    swf_engine.export_to_gui("/tmp/mulm_stats_coefficients", X=X, Y=Y)

    # 3) Prediction statistics for each block (of size k) return
    # a block_size x n_yfeatures array of t-values and p-values
    mulm_stats_predictions = ColumnSplitter(MUOLSStatsPredictions(),
                                {"X": x_group_indices})
    swf_engine = SomaWorkflowEngine(tree_root=mulm_stats_predictions,
                                    num_processes=2)
    swf_engine.export_to_gui("/tmp/mulm_stats_predictions", X=X, Y=Y)
