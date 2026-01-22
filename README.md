# House Price Prediction using Machine Learning

## Problem Statement
Predict house sale prices based on property features using machine learning regression models.

## Dataset
- Housing dataset containing numerical and categorical features
- Includes property size, location, quality, and renovation details

## Approach
- Performed data cleaning and domain-driven missing value imputation
- Conducted feature engineering (house age, total square footage, total bathrooms, etc.)
- Performed exploratory data analysis (EDA) to identify outliers and key drivers
- Trained and compared multiple regression models:
  - Linear Regression
  - Ridge, Lasso, ElasticNet
  - Decision Tree, Random Forest
  - XGBoost
- Evaluated models using MAE, RMSE, and R²
- Selected XGBoost as the final model
- Validated performance using residual analysis and feature importance

## Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- XGBoost
- Matplotlib, Seaborn

## Results
- XGBoost achieved the best performance with lowest RMSE
- Feature engineering significantly improved predictive accuracy
- Residuals showed minimal bias and good generalization

## Future Improvements
- Hyperparameter optimization with Optuna
- Add cross-validation pipelines
- Deploy model as an API
