# Vehicle Price Prediction 

This repository contains an end-to-end Machine Learning pipeline designed to predict used vehicle prices based on technical features, vehicle condition, and market attributes. It provides a modular approach to data cleaning, custom feature engineering, data preprocessing, and model evaluation to serve a robust, production-ready system.

---

## What the Project Does
* **Data Cleaning & Pipeline:** Standardizes column names, corrects extreme price and mileage outliers, removes invalid production years, and imputes missing variables (such as engine volume) using group medians based on vehicle make and model.
* **Feature Engineering:** Computes advanced interaction characteristics including vehicle age (`car_age`), average mileage driven per year (`mileage_per_year`), engine volume converted to liters (`engine_volume_liters`), and custom vehicle-wear categories.
* **Production-Grade Preprocessing:** Utilizes Scikit-Learn transformers (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `OrdinalEncoder`) to seamlessly prepare both text and numeric features for machine learning.
* **Model Serialization:** Exports the final winning pipeline architecture into a unified `joblib` file for instant inference on raw, unseen text data.

---

## Dataset
The project processes a vehicle dataset containing the following features:

* `make` / `model` / `segment` (Brand, vehicle model, and car market classification)
* `priceUSD` (Target variable)
* `year` (Year of manufacture)
* `condition` (e.g., with mileage, crashed, for parts)
* `mileage(kilometers)` (Total distance driven)
* `fuel_type` / `volume(cm3)` (Engine fuel type and displacement)
* `transmission` / `drive_unit` (Gearbox type and drivetrain layout)
* `color` (Exterior paint color)

---

## How to Run the Code

### 1. Prerequisites
Install the required dependencies using pip:
```bash
pip install pandas numpy scikit-learn joblib
```

### 2. Execution Steps
Run the modular pipeline scripts in chronological order from the project root directory:

```bash
# 1. Clean data and apply feature engineering
python src/feature_engineering.py

# 2. Run data preprocessing and train models
python src/data_preprocessing.py

# 3. Evaluate models and save the best pipeline to disk
python src/model_evaluation.py
```

---

## Tested Models & Performance Results

Four regression algorithms were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the R-squared ($R^2$) coefficient.

| Model | MAE ($) | RMSE ($) | $R^2$ (Explained Variance) |
| :--- | :---: | :---: | :---: |
| **Random Forest** | **1,071.275** | **2,674.528** | **0.899 (89.9%)** |
| Gradient Boosting | 1,587.889 | 3,185.147 | 0.857 (85.7%) |
| Decision Tree | 1,390.993 | 3,238.122 | 0.852 (85.2%) |
| Linear Regression | 2,127.879 | 4271.440 | 0.743 (74.3%) |

---

## Final Model Selection & Justification

**Random Forest** was selected as the final production model for this project due to the following decisive reasons:

1. **Superior Accuracy ($R^2 = 0.899$)**: It successfully explains **89.9%** of the variance in used vehicle pricing. It significantly outperforms the single Decision Tree (85.2%) and completely dominates Linear Regression, which struggles at 74.3%.
2. **Lowest Average Prediction Error (MAE)**: It achieves the lowest mean absolute error of **$1,071.275**, making it roughly twice as accurate on daily evaluations compared to Linear Regression ($2,127.879).
3. **Robust Outlier Control (RMSE = 2,674.528)**: Automotive markets are full of high-value outliers (luxury cars or rare trims). Random Forest obtained the lowest RMSE, proving it handles variance safely and drastically limits catastrophic over or under-predictions compared to the other models.
4. **Handling Non-Linear Relationships**: Car depreciation is highly non-linear (e.g., luxury brands lose value aggressively in their first 3 years, while economy brands lose value slowly and linearly). Tree-based ensemble learning naturally captures these complex, non-linear feature thresholds and interactions out-of-the-box.
