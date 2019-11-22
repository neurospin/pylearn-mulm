# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:00:13 CET 2019

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import numpy as np
import mulm
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smfrmla

################################################################################
# Load salary table

url = 'https://raw.github.com/neurospin/pystatsml/master/datasets/salary_table.csv'
df = pd.read_csv(url)

################################################################################
# Fit with statmodel

oneway = smfrmla.ols('salary ~ experience + education + management', df).fit()
print(oneway.summary())
aov = sm.stats.anova_lm(oneway, typ=2) # Type 2 ANOVA DataFrame
print(aov)

################################################################################
# Fit with MULM

X_df = pd.get_dummies(df.iloc[:, 1:])
X = np.asarray(X_df).astype(float)
Y = np.asarray(df.salary)[:, None].astype(float)

con_exp = np.zeros((X.shape[1], X.shape[1]))
con_exp[0, 0] = 1

con_edu = np.zeros((X.shape[1], X.shape[1]))
con_edu[[1, 2, 3], [1, 2, 3]] = 1

con_man = np.zeros((X.shape[1], X.shape[1]))
con_man[[4, 5], [4, 5]] = 1


mod = mulm.MUOLS(Y, X).fit()
tvals_exp, rawp_expt, df = mod.t_test([1, 0, 0, 0, 0, 0], pval=True, two_tailed=True)
fvals_exp, rawp_exp = mod.f_test(con_exp, pval=True)
fvals_edu, rawp_edu = mod.f_test(con_edu, pval=True)
fvals_man, rawp_man = mod.f_test(con_man, pval=True)

################################################################################
# Check

assert np.allclose(aov.loc['experience', 'F'], tvals_exp[0] ** 2)
assert np.allclose(aov.loc['experience', 'PR(>F)'], rawp_expt[0] ** 2)

assert np.allclose(aov.loc['experience', 'F'], fvals_exp[0])
assert np.allclose(aov.loc['experience', 'PR(>F)'], rawp_exp[0])

assert np.allclose(aov.loc['education', 'F'], fvals_edu[0])
assert np.allclose(aov.loc['education', 'PR(>F)'], rawp_edu[0])

assert np.allclose(aov.loc['management', 'F'], fvals_man[0])
assert np.allclose(aov.loc['management', 'PR(>F)'], rawp_man[0])

