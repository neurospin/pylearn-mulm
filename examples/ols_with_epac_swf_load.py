# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:06:51 2013

@author: jinpeng.li@cea.fr
"""

from epac import SomaWorkflowEngine
import joblib

if __name__ == "__main__":
    n_samples = 10
    n_xfeatures = 20
    n_yfeatures = 15
    x_n_groups = 3
    y_n_groups = 2
    x_group_indices = joblib.load("/tmp/x_group_indices")
    blocks_size = [np.sum(lev == x_group_indices)
                    for lev in set(x_group_indices)]
    print "Blocks size", blocks_size

    # 1) Prediction for each X block return a n_samples x n_yfeatures
    mulm = SomaWorkflowEngine.load_from_gui("/tmp/mulm")
    for leaf in mulm.walk_leaves():
        print "===============leaf.load_results()================="
        print "key =", leaf.get_key()
        tab = leaf.load_results()
        print "predictions shape == (n_samples, n_yfeatures) ?:",\
            tab["MUOLS"]['Y/pred'].shape == (n_samples, n_yfeatures)

    # 2) coeficient Statistics for each block (of size k) return
    # a block_size x n_yfeatures array of t-values and p-values
    mulm_stats_coefficients = \
        SomaWorkflowEngine.load_from_gui("/tmp/mulm_stats_coefficients")
    k = 0
    for leaf in mulm_stats_coefficients.walk_leaves():
        print "===============leaf.load_results()================="
        print "key =", leaf.get_key()
        tab = leaf.load_results()
        print "p-values shape: == (k, n_yfeatures) ?:",\
            tab["MUOLSStatsCoefficients"]['pvals'].shape == (blocks_size[k],
                                                             n_yfeatures)
        k += 1

    # 3) Prediction statistics for each block (of size k) return
    # a block_size x n_yfeatures array of t-values and p-values
    mulm_stats_predictions = \
        SomaWorkflowEngine.load_from_gui("/tmp/mulm_stats_predictions")
    for leaf in mulm_stats_predictions.walk_leaves():
        print "===============leaf.load_results()================="
        print "key =", leaf.get_key()
        tab = leaf.load_results()
        print "p-values shape: == n_yfeatures ?:",\
            tab["MUOLSStatsPredictions"]['r2'].shape == (n_yfeatures, )
