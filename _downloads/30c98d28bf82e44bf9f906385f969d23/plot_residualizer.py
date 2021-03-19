"""
Residualizer
============

Credit: E Duchesnay

Residualization of a Y data on possibly adjusted for other variables.

Suppose we have 3 variable:
- site : contains a site effect
- age: some uniform value plus some site effect
- y = -0.1 * age + site + eps

The goal is to remove the site effect while preserving the age effect.
"""

################################################################################
# Import

import numpy as np
import pandas as pd
import scipy.stats as stats
from mulm.residualizer import Residualizer
import seaborn as sns

np.random.seed(1)

################################################################################
# Dataset with site effect on age
# Before residualization on site. The association between y and age is affected
# by site.

site = np.array([-1] * 50 + [1] * 50)
age = np.random.uniform(10, 40, size=100) + 5 * site
y = -0.1 * age + site + np.random.normal(size=100)
data = pd.DataFrame(dict(y=y, age=age, site=site.astype(object)))

sns.lmplot(x="age", y="y", hue="site", data=data)

################################################################################
# Simple residualization on site. Better, but removing site effect also remove
# age effect

res_spl = Residualizer(data=data, formula_res="site")
X = res_spl.get_design_mat(data)
print("Design mat contains intercept and site:")
print(X[:5, :])
yres = res_spl.fit_transform(y[:, None], X)
data["yres"] = yres
sns.lmplot(x="age", y="yres", hue="site", data=data)

################################################################################
# Site residualization adjusted for age provides higher correlation, and
# lower stderr than simple residualization.

res_adj = Residualizer(data, formula_res="site", formula_full="age + site")
X = res_adj.get_design_mat(data)
print("Design mat contains intercept, site and age")
print(X[:5, :])
print("Residualisation contrast (intercept, site):")
print(res_adj.contrast_res)

yadj = res_adj.fit_transform(y[:, None], X)

lm_res = stats.linregress(age, yres.ravel())
lm_adj = stats.linregress(age, yadj.ravel())

np.allclose((lm_res.slope, lm_res.rvalue, lm_res.stderr),
            (-0.079187578, -0.623733003, 0.0100242219))

np.allclose((lm_adj.slope, lm_adj.rvalue, lm_adj.stderr),
            (-0.110779913, -0.7909219758, 0.00865778640))

data["yadj"] = yadj
sns.lmplot(x="age", y="yadj", hue="site", data=data)
