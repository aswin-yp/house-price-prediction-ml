# Dataset Information

This project uses the **House Prices – Advanced Regression Techniques** dataset to build and evaluate machine learning models for predicting house sale prices.

## 📌 Dataset Source
- Source: Kaggle  
- Competition: *House Prices – Advanced Regression Techniques*  
- Dataset Link: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

> ⚠️ The raw dataset is **not included in this repository** to comply with Kaggle licensing and GitHub best practices.

---

## 📄 Files Used
- `train.csv` – Contains house features along with the target variable `SalePrice`
- `test.csv` – Contains house features without target values (not used in training here)

---

## 🧾 Dataset Description
The dataset contains residential home sales data with **79 explanatory variables** describing:

- Property size and layout  
- Construction quality and condition  
- Basement and garage details  
- Neighborhood and location features  
- Sale timing and sale conditions  

**Target Variable:**
- `SalePrice` – Sale price of each house in USD

---

## 🛠 Data Preprocessing
Before modeling, the dataset was cleaned and prepared using Python:

- Handled missing values using domain-specific logic
- Filled absence-based features with `"None"` (e.g., garage, pool, basement)
- Corrected inconsistent and incorrect values
- Converted categorical features into numeric form
- Created new engineered features to improve prediction accuracy

All preprocessing logic is implemented in the `src/` directory.

---

## 🧠 Feature Engineering
Additional features were created to capture important housing characteristics:

- `Age` – Age of the house  
- `TotalSquareFoot` – Combined basement and floor area  
- `TotalBathrooms` – Total number of bathrooms  
- `TotalRooms` – Total rooms above ground  
- `TotalPorchSF` – Total porch area  
- `HouseAgeAtSale` – House age at the time of sale  
- `SinceRemodel` – Years since last remodeling  

---

## 📊 Data Usage
The processed dataset is used for:
- Exploratory Data Analysis (EDA)
- Feature correlation analysis
- Training and comparing multiple regression models
- Evaluating model performance using MAE, RMSE, and R²

---

## 🔁 Reproducibility
To reproduce the dataset setup:
1. Download `train.csv` from Kaggle
2. Place it in the appropriate local directory
3. Run the preprocessing and feature engineering scripts in order

---

## ✅ Notes
- No raw data files are committed to this repository
- Only code and documentation are version-controlled
- This ensures reproducibility and ethical data usage
