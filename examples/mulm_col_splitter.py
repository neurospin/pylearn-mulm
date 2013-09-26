# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:06:51 2013

@author: jinpeng.li@cea.fr
"""


import numpy as np
import random
from epac import LocalEngine, SomaWorkflowEngine
from epac import ColumnSplitter
from mulm import MUOLS, MUOLSStats

if __name__ == "__main__":
    n_samples = 10
    n_xfeatures = 20
    n_yfeatures = 15
    x_n_groups = 3
    y_n_groups = 2

    X = np.random.randn(n_samples, n_xfeatures)
    Y = np.random.randn(n_samples, n_yfeatures)
    x_group_indices = np.array([random.randint(0, x_n_groups)\
        for i in xrange(n_xfeatures)])

    # 1) Prediction for each X block return a n_samples x n_yfeatures
    mulm = ColumnSplitter(MUOLS(),
                          {"X": x_group_indices})
    mulm.run(X=X, Y=Y)
    for leaf in mulm.walk_leaves():
        print "===============leaf.load_results()================="
        print "key =", leaf.get_key()
        tab = leaf.load_results()
        print tab["MUOLS"]['Y/pred']

    # 2) Prediction for each X block return a n_samples x n_yfeatures
    mulm_stats = ColumnSplitter(MUOLSStats(),
                                {"X": x_group_indices})
    # ========================================
    # Signle process
    mulm_stats.run(X=X, Y=Y)
    # ========================================
    # To use local-multi porcessing
#    local_engine = LocalEngine(tree_root=mulm_stats,
#                               num_processes=2)
#    mulm_stats = local_engine.run(X=X, Y=Y)
    # ========================================
    # To use soma-workflow
#    swf_engine = SomaWorkflowEngine(tree_root=mulm_stats,
#                               num_processes=2)
#    mulm_stats = swf_engine.run(X=X, Y=Y)
    for leaf in mulm_stats.walk_leaves():
        print "===============leaf.load_results()================="
        print "key =", leaf.get_key()
        tab = leaf.load_results()
        print tab["MUOLSStats"]
