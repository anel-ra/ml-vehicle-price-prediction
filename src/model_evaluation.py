import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import (
    TARGET_COLUMN,
    split_features_and_target
)

DATA_PATH = "data/cars_cleaned_with_features.csv"
MODEL_PATH = "models/linear_regression_model.joblib"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Splitting features and target...")

X, y = split_features_and_target(df)

print("Creating the same train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
 )

print("Loading trained model...")

loaded_model = joblib.load(MODEL_PATH)


print("Making predictions...")

y_pred = loaded_model.predict(X_test)
#print(y_pred[:10])
y_pred = np.clip(y_pred, a_min=100, a_max=None)

print("Calculating regression metrics...")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

metrics = pd.DataFrame({
    "metric": ["MAE", "MSE", "RMSE", "R2"],
    "value": [mae, mse, rmse, r2],
})


print("\nRegression metrics:")
# uklanjanje naucne notacije iz tabele
pd.options.display.float_format = "{:,.2f}".format
print(metrics)

# Vraća podrazumijevani Pandas prikaz na fabrička podešavanja
pd.reset_option("display.float_format")

prediction_analysis = pd.DataFrame(
    {
        "actual_price": y_test.values,
        "predicted_price": y_pred,
    }
)
prediction_analysis["error_price"] = (
    prediction_analysis["actual_price"]
    - prediction_analysis["predicted_price"]
)
prediction_analysis["absolute_error_price"] = (
   prediction_analysis["error_price"].abs()
)
  
print("\nPrediction examples:")

print(
    prediction_analysis
    .sample(10, random_state=42)   
)


 
print("\nLargest prediction errors:")
print(
    prediction_analysis
    .sort_values("absolute_error_price", ascending=False)
    .head(10)
)