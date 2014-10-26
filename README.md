Massive univariate linear model
===============================

Provides basic features of "statmodels" (like OLS) where Y is a matrix of
many responses where many independant fit are requested.


Installation
------------
Unless you already have Numpy and Scipy installed, you need to install them:

```
$ sudo apt-get install python-numpy python-scipy
```

T-tests
-------

Then simply put pylearn-mulm in your `$PYTHONPATH`

```python
import numpy as np
import mulm
import statsmodels.api as sm

n = 100
X = np.random.randn(n, 5)
Y = np.random.randn(n, 10)
beta = np.random.randn(5, 1)
# Causal model: add X on the 2 first variables
Y[:, :2] += np.dot(X, beta)

# t-test all the regressors (by default mulm and sm do two-tailed tests)
contrasts = np.identity(X.shape[1])

## OLS with statmodels, need to iterate over Y columns
sm_tvals = list()
sm_pvals = list()
for j in xrange(Y.shape[1]):
    mod = sm.OLS(Y[:, j], X)
    sm_ttest = mod.fit().t_test(contrasts)
    sm_tvals.append(sm_ttest.tvalue)
    sm_pvals.append(sm_ttest.pvalue)

sm_tvals = np.asarray(sm_tvals).T
sm_pvals = np.asarray(sm_pvals).T

## OLS with MULM
mod = mulm.MUOLS(Y, X).fit()
mulm_tvals, mulm_pvals, mulm_df = mod.t_test(contrasts, pval=True, two_tailed=True)

# Check that results ar similar
np.allclose(mulm_tvals, sm_tvals)
np.allclose(mulm_pvals, sm_pvals)
```

Multiple comparison: maxT
-------------------------


```python
import numpy as np
import mulm
import pylab as plt

n = 100
px = 5
py_info = 2
py_noize = 100

beta = np.array([1, 0, .5] + [0] * (px - 4) + [2]).reshape((px, 1))
X = np.hstack([np.random.randn(n, px-1), np.ones((n, 1))]) # X with intercept
Y = np.random.randn(n, py_info + py_noize)
# Causal model: add X on the first py_info variable
Y[:, :py_info] += np.dot(X, beta)

# t-test all the regressors (by default mulm and sm do two-tailed tests)
contrasts = np.identity(X.shape[1])

mod = mulm.MUOLS(Y, X).fit()
tvals, rawp, df = mod.t_test(contrasts, pval=True, two_tailed=True)
tvals, maxT, df2 = mod.t_test_maxT(contrasts, two_tailed=True)


n, bins, patches = plt.hist([rawp[0,:], maxT[0,:]],
                            color=['blue', 'red'], label=['rawp','maxT'])
plt.legend()
plt.show()
```

