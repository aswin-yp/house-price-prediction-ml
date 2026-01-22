import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R2:", r2_score(y_test, preds))

    residuals = y_test - preds

    plt.figure(figsize=(8,5))
    sns.scatterplot(x=preds, y=residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residual Plot")
    plt.show()
