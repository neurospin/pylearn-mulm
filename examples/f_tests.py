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
from collections import OrderedDict

########################################################################################################################
# Example 1: single target column: salary table: salary ~ experience + education + management
# -------------------------------------------------------------------------------------------

url = 'https://github.com/duchesnay/pystatsml/raw/master/datasets/salary_table.csv'
df = pd.read_csv(url)


url = 'https://stats.idre.ucla.edu/stat/data/hsb2.csv'
hsb2 = pd.read_csv(url)
########################################################################################################################

df.salary /= 100

df.groupby('education')['salary'].mean()
df['salary'].mean()

""""
education
Bachelor    149.415000
Master      182.863684
Ph.D        182.928462

172.70195652173913
"""


# 'salary', 'experience', 'education', 'management'
from statsmodels.formula.api import ols
mod = ols("salary ~ C(education, Treatment)", data=df).fit()
print(mod.summary())

mod = ols("salary ~ C(education, Sum)", data=df).fit()
print(mod.summary())

import patsy
#formula = "write ~ C(race, Treatment)"
formula = "education"

dmat = patsy.dmatrix("education", data=df)
dmat = patsy.dmatrix("C(education, Treatment)", data=df)
dmat = patsy.dmatrix("C(education, Sum)", data=df)

np.asarray(dmat)

########################################################################################################################
# Fit with MULM

Y = np.asarray(df.salary)[:, None].astype(float)
X, t_contrasts, f_contrasts = mulm.design_matrix(formula="experience + education + management", data=df)
mod_mulm = mulm.MUOLS(Y, X).fit()
aov_mulm = OrderedDict((term, mod_mulm.f_test(f_contrasts[term], pval=True)) for term in f_contrasts)

print(mod_mulm.coef)
print(aov_mulm)


########################################################################################################################
# Fit with statsmodel

mod_sm = smfrmla.ols('salary ~ experience + education + management', df).fit()
aov_sm = sm.stats.anova_lm(mod_sm, typ=2) # Type 2 ANOVA DataFrame
print(mod_sm.summary())
print(aov_sm)

# Check equality of model parameters
assert np.allclose(np.asarray(mod_sm.params), mod_mulm.coef[:, 0])

########################################################################################################################
# Check equality of F-statistics

assert np.allclose(np.asarray(aov_sm.loc[:, "F"][:-1]),
    np.asarray([aov_mulm[iv][0][0] for iv in list(f_contrasts.keys())[1:]]))

########################################################################################################################
# Check equality of P-values
assert np.allclose(np.asarray(aov_sm.loc[:, "PR(>F)"][:-1]),
    np.asarray([aov_mulm[iv][1][0] for iv in list(f_contrasts.keys())[1:]]))

########################################################################################################################
# Example 2: Multiple targets: y_i = age + sex + site
# ---------------------------------------------------

########################################################################################################################
# Build dataset
age = np.random.normal(size=100)
sex = np.random.choice([0, 1], 100)
sex_c = ["X%i" % i for i in sex]
site = np.array([2] * 25 + [1] * 25 + [-1] * 50)
site_c = ["S%i" % i for i in site]

# Independent variables
x_df = pd.DataFrame(OrderedDict(age=age, sex=sex_c, site=site_c))

# Dependent variables
y_dict = OrderedDict(
    y0 = 0.1 * age + 0.1 * sex + site + np.random.normal(size=100), # age and sex
    y1 = 0.1 * age + 0.0 * sex + site + np.random.normal(size=100), # age only
    y2 = 0.0 * age + 0.1 * sex + site + np.random.normal(size=100)) # sex only
for i in range(3, 10):
    y_dict["y%i" % i] = 0.0 * age + 0.0 * sex + site + np.random.normal(size=100)

y_df = pd.DataFrame(y_dict)
Y = np.asarray(y_df)

data = pd.concat((y_df, x_df), axis=1)

########################################################################################################################
# Fit with MULM

X, t_contrasts, f_contrasts = mulm.design_matrix(formula="age + sex + site", data=x_df)
mod_mulm = mulm.MUOLS(Y, X).fit()
aov_mulm = OrderedDict((term, mod_mulm.f_test(f_contrasts[term], pval=True)) for term in f_contrasts)

########################################################################################################################
# With statsmodels

import statsmodels.api as sm
import statsmodels.formula.api as smfrmla

mod_sm = {dv:smfrmla.ols("%s ~ %s" % (dv, "age + sex + site"), data=data).fit() for dv in y_df.columns}

########################################################################################################################
# Check equality of model parameters
for idx, dv in enumerate(y_df.columns):
    assert np.allclose(np.asarray(mod_sm[dv].params), mod_mulm.coef[:, idx])

########################################################################################################################
# Check equality of F-statistics
aov_sm = {dv:sm.stats.anova_lm(mod_sm[dv], typ=2) for dv in mod_sm}

for idx, dv in enumerate(y_df.columns):
    assert np.allclose(np.asarray(aov_sm[dv].loc[:, "F"][:-1]),
        np.asarray([aov_mulm[iv][0][idx] for iv in list(f_contrasts.keys())[1:]]))

########################################################################################################################
# Check equality of P-values
for idx, dv in enumerate(y_df.columns):
    assert np.allclose(np.asarray(aov_sm[dv].loc[:, "PR(>F)"][:-1]),
        np.asarray([aov_mulm[iv][1][idx] for iv in list(f_contrasts.keys())[1:]]))
