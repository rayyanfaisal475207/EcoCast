import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

# Load data
data_path = 'd:/ML/New folder/data/cleaned_data_ml.csv'
df = pd.read_csv(data_path)

# Shuffle
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Feature columns
if 'AQI_lag1' in df.columns:
    aqi_lag1_col = 'AQI_lag1'
elif 'aqilag1' in df.columns:
    aqi_lag1_col = 'aqilag1'
else:
    cols = df.columns.tolist()
    aqi_lag1_col = next((c for c in cols if 'aqi' in c.lower() and 'lag1' in c.lower()), None)

if 'AQI_roll3' in df.columns:
    aqi_roll3_col = 'AQI_roll3'
elif 'aqiroll3' in df.columns:
    aqi_roll3_col = 'aqiroll3'
else:
    cols = df.columns.tolist()
    aqi_roll3_col = next((c for c in cols if 'aqi' in c.lower() and 'roll3' in c.lower()), None)

feature_cols = [
    'PM2.5', 'NO', 'CO', 'SO2', 'O3',
    'Day', 'Month', 'Weekday',
    'PM2.5_roll3', 'NO_roll3',
    'CO_lag1', 'CO_roll3',
    'SO2_roll3', 'O3_roll3',
    'Season_enc',
    'PM2.5_cityavg', 'NO_cityavg', 'CO_cityavg',
    'SO2_cityavg', 'O3_cityavg',
    'City_enc',
    'Temperature', 'Humidity', 'Wind_speed', 'Rainfall',
    aqi_lag1_col, aqi_roll3_col
]

X = df_shuffled[feature_cols].values
y = df_shuffled['AQI'].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
xgb_model.fit(X_train, y_train)

# Evaluate
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")

# Save model
model_path = 'd:/ML/New folder/xgboost_model.pkl'
joblib.dump(xgb_model, model_path)

# City Mapping (Alphabetical based on LabelEncoder behavior)
city_names_list = [
    "Faisalabad", "Gujranwala", "Hyderabad", "Islamabad", "Karachi",
    "Lahore", "Multan", "Peshawar", "Quetta", "Rawalpindi"
]
# Assuming 0-9 maps to this sorted list

# Extract Metadata
city_meta = {}
city_groups = df.groupby('City_enc')

for city_enc, group in city_groups:
    enc_int = int(city_enc)
    name = city_names_list[enc_int] if enc_int < len(city_names_list) else f"City {enc_int}"
    
    meta = {
        'name': name,
        'PM2.5_cityavg': float(group['PM2.5_cityavg'].iloc[0]),
        'NO_cityavg': float(group['NO_cityavg'].iloc[0]),
        'CO_cityavg': float(group['CO_cityavg'].iloc[0]),
        'SO2_cityavg': float(group['SO2_cityavg'].iloc[0]),
        'O3_cityavg': float(group['O3_cityavg'].iloc[0]),
        'count': int(len(group))
    }
    city_meta[enc_int] = meta

metadata = {
    'feature_names': feature_cols,
    'city_metadata': city_meta,
    'aqi_lag1_col': aqi_lag1_col,
    'aqi_roll3_col': aqi_roll3_col,
    'model_accuracy': {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2)
    }
}

with open('d:/ML/New folder/city_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("Metadata updated with city names and accuracy.")
