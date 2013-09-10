# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 09:13:14 2013

@author: edouard
"""
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
import mulm

class TestMULMOLS(unittest.TestCase):

    def test_regression(self):
        X = np.array([[ 0.65755752,  0.28764865,  0.57877508,  0.90387418,  1.],
               [ 0.79048795, -1.16186096, -1.12557116, -0.61673564,  1.],
               [-0.31715617, -0.98712971, -1.01510942,  0.28110482,  1.],
               [-1.60268282,  1.45435033,  1.59554253, -0.90958868,  1.],
               [-1.39936974, -0.48804216, -0.29716962,  0.15797782,  1.],
               [-0.5016932 ,  0.63878573,  0.50386062,  1.00779904,  1.],
               [-0.58616703,  0.50147743,  0.40283611, -0.34769953,  1.],
               [-1.1801445 ,  0.57331598,  0.45630847,  0.11252948,  1.],
               [-1.01391773,  0.54845523,  0.34128012,  0.35652613,  1.],
               [ 1.11852204, -0.87827513, -1.05929458, -0.58185942,  1.]])

        Y = np.array([[ 6.35423611,  4.0702969 ,  1.24435593, -0.38972197],
               [-2.26824906, -0.87068743, -0.91946825, -0.1533058 ],
               [-3.05580016, -1.52189299,  0.81468903, -0.55042916],
               [ 7.72146505,  6.09847529,  1.86804166, -0.51907661],
               [-0.71992246, -0.43137279, -0.16284335, -0.54793912],
               [ 6.13244807,  5.54908003, -1.56168587,  0.36359084],
               [ 3.56813057,  2.54402591, -0.89267959, -0.80318265],
               [ 3.36148481,  4.04922236,  1.17058636,  0.60791983],
               [ 2.6606488 ,  3.51918414,  1.14627325, -0.19035388],
               [-0.48922081,  0.70832416,  0.52489858,  1.84431222]])
        # Fit
        betas, ss_errors = mulm.ols(X, Y)
        fmaps = list()
        tmaps = list()
        pfmaps = list()
        ptmaps = list()
        # Test compute all 1D contrasts
        for i in xrange(5):
            contrast = [0]*i + [1] + [0]*(5-i-1)
            #print contrast
            f, pf = mulm.ols_stats_fcon(X, betas, ss_errors, contrast, pval=True)
            t, pt = mulm.ols_stats_tcon(X, betas, ss_errors, contrast, pval=True)
            fmaps.append(f)
            pfmaps.append(pf)
            tmaps.append(t)
            ptmaps.append(pt)
        fmaps = np.asarray(fmaps)
        tmaps = np.asarray(tmaps)
        pfmaps = np.asarray(pfmaps)
        ptmaps = np.asarray(ptmaps)
        ## Check t^2 = F
        assert_almost_equal(tmaps ** 2, fmaps, decimal=10)
        ## Where beta < 0 t contrast p-value = 1 - p-value
        ptmaps[betas < 0] = 1 - ptmaps[betas < 0]
        ## Check F contrast p-values / 2 == F contrast p-values
        assert_almost_equal(pfmaps / 2, ptmaps, decimal=10)
        # Check versus results obtained in R
        betas_r = \
        np.array([[ 1.54304841,  0.93608702, -0.06212768,  0.41439447],
               [ 1.58512552,  4.31956633, -1.20671898,  2.42143663],
               [ 3.46108808, -0.75049613,  1.5728053 , -2.32513322],
               [ 0.56219508,  0.41220736, -0.32993793, -0.09236295],
               [ 2.71911979,  2.55165455,  0.30913782,  0.10708515]])
        tmaps_r = \
        np.array([[  5.41092398,   2.86631787,  -0.10042642,   1.20341298],
               [  1.21894398,   2.90052412,  -0.42775703,   1.5420628 ],
               [  2.74536542,  -0.51981907,   0.5750868 ,  -1.52736984],
               [  1.67813501,   1.07441445,  -0.4539866 ,  -0.22832134],
               [ 11.82453133,   9.68932935,   0.61969595,   0.38565033]])
        ptmaps_r = \
        np.array([[  2.91639375e-03,   3.51456477e-02,   9.23908342e-01,
                  2.82680677e-01],
               [  2.77230919e-01,   3.37700852e-02,   6.86625661e-01,
                  1.83698283e-01],
               [  4.05327476e-02,   6.25374986e-01,   5.90141455e-01,
                  1.87202389e-01],
               [  1.54161654e-01,   3.31728227e-01,   6.68865097e-01,
                  8.28440359e-01],
               [  7.61495629e-05,   1.98841407e-04,   5.62602711e-01,
                  7.15618016e-01]])
        # betas vs those in R
        assert_almost_equal(betas, betas_r, decimal=7)
        assert_almost_equal(tmaps, tmaps_r, decimal=6)
        assert_almost_equal(pfmaps, ptmaps_r, decimal=7)

if __name__ == '__main__':
    unittest.main()