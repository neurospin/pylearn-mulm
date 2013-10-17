# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:06:51 2013

@author: jinpeng.li@cea.fr
"""

from epac import SomaWorkflowEngine
import joblib
import numpy as np

if __name__ == "__main__":

    ols_config = joblib.load("/tmp/example_config")
    blocks_size = [np.sum(lev == np.asarray(ols_config.x_group_indices))
                    for lev in set(ols_config.x_group_indices)]
    print "Blocks size", blocks_size

    # 1) Prediction for each X block return a n_samples x n_yfeatures
    mulm = SomaWorkflowEngine.load_from_gui("/tmp/mulm")
    res_tab = mulm.reduce()
    for res_tab_key in res_tab.keys():
        print "===============key=%s=============" % res_tab_key
        print "predictions shape == (n_samples, n_yfeatures) ?:", \
              res_tab[res_tab_key]['Y/pred'].shape == (ols_config.n_samples,
                                                       ols_config.n_yfeatures)

    # 2) coeficient Statistics for each block (of size k) return
    # a block_size x n_yfeatures array of t-values and p-values
    mulm_stats_coefficients = \
        SomaWorkflowEngine.load_from_gui("/tmp/mulm_stats_coefficients")
    res_tab = mulm_stats_coefficients.reduce()
    k = 0
    for res_tab_key in res_tab.keys():
        print "===========key=%s============" % res_tab_key
        print "p-values shape: == (k, n_yfeatures) ?:",\
               res_tab[res_tab_key]['pvals'].shape == \
                                (blocks_size[k], ols_config.n_yfeatures)
        k += 1

    # 3) Prediction statistics for each block (of size k) return
    # a block_size x n_yfeatures array of t-values and p-values
    mulm_stats_predictions = \
        SomaWorkflowEngine.load_from_gui("/tmp/mulm_stats_predictions")
    res_tab = mulm_stats_predictions.reduce()
    for res_tab_key in res_tab.keys():
        print "===========key=%s============" % res_tab_key
        print "p-values shape: == n_yfeatures ?:",\
            res_tab[res_tab_key]['r2'].shape == (ols_config.n_yfeatures, )
