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


class ExampleConfig:
    def __init__(self):
        self.n_samples = 10
        self.n_xfeatures = 20
        self.n_yfeatures = 15
        self.x_n_groups = 3
        self.y_n_groups = 2
        self.x_group_indices = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3,
                                4, 4, 4, 4, 5, 5, 5]
#        self.x_group_indices = np.array([
#            random.randint(0, self.x_n_groups)\
#            for i in xrange(self.n_xfeatures)])

if __name__ == "__main__":
    ols_config = ExampleConfig()
    X = np.random.randn(ols_config.n_samples, ols_config.n_xfeatures)
    Y = np.random.randn(ols_config.n_samples, ols_config.n_yfeatures)
    joblib.dump(ols_config, "/tmp/example_config")

    blocks_size = [np.sum(lev == ols_config.x_group_indices)
                    for lev in set(ols_config.x_group_indices)]
    print "Blocks size", blocks_size

    # 1) Prediction for each X block return a n_samples x n_yfeatures
    mulm = ColumnSplitter(MUOLS(),
                          {"X": ols_config.x_group_indices})
    swf_engine = SomaWorkflowEngine(tree_root=mulm,
                                    num_processes=2)
    swf_engine.export_to_gui("/tmp/mulm", X=X, Y=Y)

    # 2) coeficient Statistics for each block (of size k) return
    # a block_size x n_yfeatures array of t-values and p-values
    mulm_stats_coefficients = ColumnSplitter(MUOLSStatsCoefficients(),
                                {"X": ols_config.x_group_indices})
    swf_engine = SomaWorkflowEngine(tree_root=mulm_stats_coefficients,
                                    num_processes=2)
    swf_engine.export_to_gui("/tmp/mulm_stats_coefficients", X=X, Y=Y)

    # 3) Prediction statistics for each block (of size k) return
    # a block_size x n_yfeatures array of t-values and p-values
    mulm_stats_predictions = ColumnSplitter(MUOLSStatsPredictions(),
                                {"X": ols_config.x_group_indices})
    swf_engine = SomaWorkflowEngine(tree_root=mulm_stats_predictions,
                                    num_processes=2)
    swf_engine.export_to_gui("/tmp/mulm_stats_predictions", X=X, Y=Y)
