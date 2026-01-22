import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression())
        ]),
        "RidgeCV": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3,3,25), cv=5))
        ]),
        "LassoCV": Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(alphas=np.logspace(-4,1,50), cv=5, max_iter=5000))
        ]),
        "ElasticNetCV": Pipeline([
            ("scaler", StandardScaler()),
            ("en", ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9],
                                alphas=np.logspace(-4,1,50),
                                cv=5, max_iter=5000))
        ]),
        "DecisionTree": Pipeline([
            ("scaler", StandardScaler()),
            ("dt", DecisionTreeRegressor(random_state=42))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                eval_metric="logloss"
            ))
        ])
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        })

    return pd.DataFrame(results).sort_values("RMSE"), models
