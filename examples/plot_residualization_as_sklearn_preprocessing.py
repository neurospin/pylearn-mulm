"""
Residualizer as pre-processing of supervized prediction:

Input: X = age + site + e, target = age

Preprocessing:
- Residualize X for "site" adjusted for "age"
- Learn to predict age on residualized data

Since age is used in residualization, it MUST be fitted on training data only.
"""

from sklearn import linear_model
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn import metrics

################################################################################
# Dataset
# X: input data of the predictive model and y is the target

site = np.array([-1] * 50 + [1] * 50)
age = np.random.uniform(10, 40, size=100) + 5 * site
X = np.random.randn(100, 5)
X[:, 0] = -0.1 * age + site + np.random.normal(size=100)
X[:, 1] = -0.1 * age + site + np.random.normal(size=100)
demographic_df = pd.DataFrame(dict(age=age, site=site.astype(object)))
y = age

################################################################################
# Predictive model cross-validation

lr = linear_model.Ridge(alpha=1)
scaler = StandardScaler()
cv = KFold(n_splits=5, random_state=42)

################################################################################
# Usage 1: Manual slicing of train/test data: use `Residualizer`

residualizer = Residualizer(data=demographic_df, formula_res='site',
                            formula_full='site + age')
Z = residualizer.get_design_mat()
scores = np.zeros((5, 2))
for i, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
    X_tr, X_te = X[tr_idx, :], X[te_idx, :]
    Z_tr, Z_te = Z[tr_idx, :], Z[te_idx, :]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # 1) Fit residualizer
    residualizer.fit(X_tr, Z_tr)

    # 2) Residualize
    X_res_tr = residualizer.transform(X_tr, Z_tr)
    X_res_te = residualizer.transform(X_te, Z_te)

    X_res_tr = scaler.fit_transform(X_res_tr)
    X_res_te = scaler.transform(X_res_te)

    # 3) Fit predictor on train residualized data
    lr.fit(X_res_tr, y_tr)

    # 4) Predict on test residualized data
    y_test_pred = lr.predict(X_res_te)

    # 5) Compute metrics
    scores[i, 0] = metrics.r2_score(y_te, y_test_pred)
    scores[i, 1] = metrics.mean_absolute_error(y_te, y_test_pred)

scores = pd.DataFrame(scores, columns=['r2', 'mae'])

print("Mean scores")
print(scores.mean(axis=0))


################################################################################
# Usage 2: Usage with sklearn pipeline, cross_validate: use `ResidualizerEstimator`

from mulm.residualizer import ResidualizerEstimator

residualizer = Residualizer(data=demographic_df, formula_res='site',
                            formula_full='site + age')
# Extract design matrix and pack it with X
Z = residualizer.get_design_mat()

residualizer_wrapper = ResidualizerEstimator(residualizer)
ZX = residualizer_wrapper.pack(Z, X)

pipeline = make_pipeline(residualizer_wrapper, StandardScaler(), lr)
cv_res = cross_validate(estimator=pipeline, X=ZX, y=y, cv=cv, n_jobs=5,
                        scoring=['r2', 'neg_mean_absolute_error'])

r2 = cv_res['test_r2'].mean()
mae = np.mean(-cv_res['test_neg_mean_absolute_error'])
print("CV R2:%.4f, MAE:%.4f" % (r2, mae))
assert np.allclose(scores.mean(axis=0).values, np.array([r2, mae]))
