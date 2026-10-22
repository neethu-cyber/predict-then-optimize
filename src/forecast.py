"""Demand forecasting: a naive baseline, linear regression, gradient boosting (point) and quantile gradient boosting.

All models are trained per medicine. `fit_predict` trains on `train_mask` rows and predicts `test_mask` rows.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from data import features

LINEAR_COLS = ["dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "season_sin", "season_cos", "trend",
               "flu_lag1", "flu_lag7", "demand_lag1", "demand_lag7", "demand_mean7", "demand_mean28"]
TREE_COLS = ["dow", "season_sin", "season_cos", "trend", "flu_lag1", "flu_lag7", "demand_lag1", "demand_lag7",
             "demand_mean7", "demand_mean28"]


def gbm(**kw):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20, random_state=0, **kw)


def fit_predict(df, med, train_mask, test_mask, quantile=None):
    """Return a DataFrame of predictions for the test rows: naive, linear, gbm and (optionally) gbm_quantile."""
    X = features(df, med)
    y = df[med].astype(float)
    tr = train_mask & X.notna().all(axis=1)
    te = test_mask
    out = pd.DataFrame(index=df.index[te])
    out["actual"] = y[te]
    out["naive"] = X.loc[te, "demand_lag7"]  # same weekday last week
    out["linear"] = np.clip(LinearRegression().fit(X.loc[tr, LINEAR_COLS], y[tr]).predict(X.loc[te, LINEAR_COLS]), 0, None)
    out["gbm"] = np.clip(gbm().fit(X.loc[tr, TREE_COLS], y[tr]).predict(X.loc[te, TREE_COLS]), 0, None)
    if quantile is not None:
        model = gbm(loss="quantile", quantile=quantile).fit(X.loc[tr, TREE_COLS], y[tr])
        out["gbm_quantile"] = np.clip(model.predict(X.loc[te, TREE_COLS]), 0, None)
    return out
