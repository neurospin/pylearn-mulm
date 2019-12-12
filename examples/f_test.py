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

mod_sm = smfrmla.ols('salary ~ experience + education + management', df).fit()
aov_sm = sm.stats.anova_lm(oneway, typ=2) # Type 2 ANOVA DataFrame
print(mod_sm.summary())
print(aov_sm)

################################################################################
# Fit with MULM
Y = np.asarray(df.salary)[:, None].astype(float)
X, t_contrasts, f_contrasts = mulm.design_matrix(formula="experience + education + management", data=df)
mod_mulm = mulm.MUOLS(Y, X).fit()
aov_mulm = OrderedDict((term, mod_mulm.f_test(f_contrasts[term], pval=True)) for term in f_contrasts)

print(mod_mulm.coef)
print(aov_mulm)

# Check equality of model parameters
assert np.allclose(np.asarray(mod_sm.params), mod.coef[:, 0])

################################################################################
# Check equality of F-statistics

assert np.allclose(np.asarray(aov_sm.loc[:, "F"][:-1]),
    np.asarray([aov_mulm[iv][0][0] for iv in list(f_contrasts.keys())[1:]]))

# Check equality of P-values
assert np.allclose(np.asarray(aov_sm.loc[:, "PR(>F)"][:-1]),
    np.asarray([aov_mulm[iv][1][0] for iv in list(f_contrasts.keys())[1:]]))

