import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Directory containing Navier-Stokes output CSV files
output_dir = "data/ns_output/"
xgb_output_dir = "data/xgb_output/"
os.makedirs(xgb_output_dir, exist_ok=True)

files = sorted(os.listdir(output_dir))

# Aggregate data
data_frames = []

# Process each CSV file
for file in files:
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(output_dir, file))
        df["wind_speed"] = np.sqrt(df["wind_x"]**2 + df["wind_y"]**2)
        data_frames.append(df)

# Combine data from all files
df = pd.concat(data_frames, ignore_index=True)

# Identify low-wind areas (potential snow drift locations)
low_wind_threshold = df["wind_speed"].quantile(0.2)  # Bottom 20% of wind speeds
df["snow_drift_risk"] = df["wind_speed"] < low_wind_threshold

# Feature Engineering
features = ["x", "y", "wind_speed", "pressure"]
target = "snow_drift_risk"

# Handle missing values (if any)
df = df.dropna()
X = df[features]
y = df[target].astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict on full dataset
df["predicted_snow_drift_risk"] = model.predict(X)

# Save results
output_file = os.path.join(xgb_output_dir, "snow_drift_predictions.csv")
df.to_csv(output_file, index=False)
print(f"Snow drift predictions saved to {output_file}")

# Save the trained model
model.save_model(os.path.join(xgb_output_dir, "snow_drift_xgboost.json"))
print("XGBoost model saved to data/xgb_output/snow_drift_xgboost.json")
