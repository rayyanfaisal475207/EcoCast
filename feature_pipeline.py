import pandas as pd
import numpy as np
import json
from datetime import datetime

class FeaturePipeline:
    def __init__(self, metadata_path='city_metadata.json'):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.city_meta = self.metadata['city_metadata']
        self.feature_names = self.metadata['feature_names']
        
        # Hardcoded scaling parameters (Estimated based on typical data and scaled range)
        # Since original scaler is missing.
        self.scaling_params = {
            'PM2.5': {'mean': 100.0, 'std': 80.0},
            'NO': {'mean': 25.0, 'std': 20.0},
            'CO': {'mean': 1.0, 'std': 1.0},
            'SO2': {'mean': 10.0, 'std': 8.0},
            'O3': {'mean': 30.0, 'std': 20.0}
        }
        
    def get_season(self, month):
        # 12,1,2 Winter(3); 3,4,5 Summer(2); 6,7,8 Monsoon(0); 9,10,11 Post-Monsoon(1)
        if month in [12, 1, 2]: return 3
        elif month in [3, 4, 5]: return 2
        elif month in [6, 7, 8]: return 0
        else: return 1

    def estimate_aqi(self, pm25, no, co, so2, o3):
        # Estimate AQI from RAW values
        p_idx = pm25 * 2
        co_idx = co * 10
        so2_idx = so2 * 2
        o3_idx = o3 * 2
        return max(p_idx, co_idx, so2_idx, o3_idx, 50)

    def scale_value(self, col, value):
        if col in self.scaling_params:
            params = self.scaling_params[col]
            return (value - params['mean']) / params['std']
        return value

    def preprocess(self, input_data):
        # 1. Basic Features
        city_enc = str(input_data['City_enc'])
        if city_enc not in self.city_meta:
            city_enc = list(self.city_meta.keys())[0]
            
        city_stats = self.city_meta[city_enc]
        
        date = input_data.get('Date', datetime.now())
        day = date.day
        month = date.month
        weekday = date.weekday()
        
        season_enc = self.get_season(month)
        
        # 2. Pollutants (Raw)
        pm25_raw = input_data['PM2.5']
        no_raw = input_data['NO']
        co_raw = input_data['CO']
        so2_raw = input_data['SO2']
        o3_raw = input_data['O3']
        
        # 3. Weather (Raw - Model seems to use raw/weirdly scaled Temp/Hum/Wind/Rain)
        # Based on stats, Temp/Hum are raw. Wind/Rain are raw (or unknown transform).
        temp = input_data['Temperature']
        hum = input_data['Humidity']
        ws = input_data['Wind_speed']
        rain = input_data['Rainfall']
        
        # 4. Scale Pollutants for Model
        pm25_scaled = self.scale_value('PM2.5', pm25_raw)
        no_scaled = self.scale_value('NO', no_raw)
        co_scaled = self.scale_value('CO', co_raw)
        so2_scaled = self.scale_value('SO2', so2_raw)
        o3_scaled = self.scale_value('O3', o3_raw)
        
        # 5. Derived/Lag Features
        # Use SCALED values for pollutant lags/rolls
        pm25_roll3 = pm25_scaled
        no_roll3 = no_scaled
        co_lag1 = co_scaled
        co_roll3 = co_scaled
        so2_roll3 = so2_scaled
        o3_roll3 = o3_scaled
        
        # Use RAW estimated AQI for AQI lags (since AQI in training data is unscaled)
        estimated_aqi = self.estimate_aqi(pm25_raw, no_raw, co_raw, so2_raw, o3_raw)
        aqi_lag1 = estimated_aqi
        aqi_roll3 = estimated_aqi
        
        # 6. City Averages (Already scaled in metadata)
        pm25_cityavg = city_stats['PM2.5_cityavg']
        no_cityavg = city_stats['NO_cityavg']
        co_cityavg = city_stats['CO_cityavg']
        so2_cityavg = city_stats['SO2_cityavg']
        o3_cityavg = city_stats['O3_cityavg']
        
        # 7. Construct DataFrame
        data = {
            'PM2.5': [pm25_scaled],
            'NO': [no_scaled],
            'CO': [co_scaled],
            'SO2': [so2_scaled],
            'O3': [o3_scaled],
            'Day': [day],
            'Month': [month],
            'Weekday': [weekday],
            'PM2.5_roll3': [pm25_roll3],
            'NO_roll3': [no_roll3],
            'CO_lag1': [co_lag1],
            'CO_roll3': [co_roll3],
            'SO2_roll3': [so2_roll3],
            'O3_roll3': [o3_roll3],
            'Season_enc': [season_enc],
            'PM2.5_cityavg': [pm25_cityavg],
            'NO_cityavg': [no_cityavg],
            'CO_cityavg': [co_cityavg],
            'SO2_cityavg': [so2_cityavg],
            'O3_cityavg': [o3_cityavg],
            'City_enc': [int(city_enc)],
            'Temperature': [temp],
            'Humidity': [hum],
            'Wind_speed': [ws],
            'Rainfall': [rain],
            self.metadata['aqi_lag1_col']: [aqi_lag1],
            self.metadata['aqi_roll3_col']: [aqi_roll3]
        }
        
        df = pd.DataFrame(data)
        df = df[self.feature_names]
        
        return df
