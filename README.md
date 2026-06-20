<div align="center">

# 🌍 EcoCast Intelligence

### Predicting Air Quality Index & Health Risk in Major Pakistani Cities Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-006400?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-9ACD32?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

[![Made with Pandas](https://img.shields.io/badge/Made%20with-Pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Visualizations-Plotly-3F4F75.svg?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)

**A production-ready ML system forecasting next-day AQI for 10 major Pakistani cities, achieving R² = 0.816 with XGBoost.**

[Overview](#-overview) •
[Results](#-results) •
[Installation](#-installation) •
[Usage](#-usage) •
[Project Structure](#-project-structure) •
[Demo](#-demo)

</div>

---

## 📖 Overview

Air pollution is one of Pakistan's most pressing public health crises, with cities like **Lahore**, **Faisalabad**, and **Karachi** routinely ranking among the world's most polluted urban centers. **EcoCast Intelligence** addresses the lack of proactive, predictive air quality infrastructure in Pakistan by combining heterogeneous, multi-source environmental data with ensemble machine learning to forecast **next-day Air Quality Index (AQI)** values — enabling timely public health interventions.

This repository contains the full pipeline: data integration, feature engineering, model training/evaluation, and an interactive **Streamlit** web application for real-time predictions and health risk advisories.

> 📄 Developed as part of a course project at the **National University of Sciences and Technology (NUST)**, Department of Computing, SEECS.

### Cities Covered

`Faisalabad` · `Islamabad` · `Lahore` · `Karachi` · `Gujranwala` · `Hyderabad` · `Multan` · `Peshawar` · `Quetta` · `Rawalpindi`

### ✨ Key Features

- 🔗 **Multi-Source Data Integration** — Unified dataset built from IQAir, OpenAQ API, provincial EPA dashboards, and Kaggle archives (2020–2025)
- 🧪 **Rigorous Feature Engineering** — Lag features, rolling averages, city-level aggregations, seasonal encodings, and high-AQI event flags
- 🤖 **7 ML Models Benchmarked** — Linear, Ridge, Lasso, Random Forest (tuned/untuned), XGBoost (3 variants), and LightGBM
- 📊 **Best Model: XGBoost (Shuffled)** — Test R² = **0.816**, MAE = **49.06**, RMSE = **65.75**
- 🏥 **Health Risk Framework** — EPA-aligned AQI categorization with population-specific health advisories
- 🖥️ **Interactive Web App** — Real-time predictions, historical trends, city comparisons, and feature importance dashboards via Streamlit + Plotly

---

## 📊 Results

### Model Performance Comparison

| Model                     | Train MAE | Train RMSE | Train R²  | Test MAE  | Test RMSE | Test R²   |
| ------------------------- | --------- | ---------- | --------- | --------- | --------- | --------- |
| Linear Regression         | 57.50     | 72.54      | 0.765     | 57.13     | 72.27     | 0.765     |
| Ridge Regression          | 57.48     | 72.51      | 0.766     | 57.12     | 72.25     | 0.766     |
| Lasso Regression          | 57.52     | 72.58      | 0.764     | 57.15     | 72.31     | 0.764     |
| Random Forest (Untuned)   | 18.71     | 25.28      | 0.972     | 50.32     | 67.38     | 0.796     |
| Random Forest (Tuned)     | 40.69     | 54.62      | 0.867     | 51.94     | 69.34     | 0.783     |
| XGBoost (Standard)        | 39.25     | 52.84      | 0.876     | 49.02     | 65.80     | 0.805     |
| **XGBoost (Shuffled)** 🏆 | **38.92** | **52.44**  | **0.878** | **49.06** | **65.75** | **0.816** |
| XGBoost (Log-Transform)   | 40.15     | 53.87      | 0.871     | 50.24     | 67.12     | 0.798     |
| LightGBM                  | 42.64     | 57.25      | 0.854     | 49.13     | 65.71     | 0.806     |

### Error by AQI Category (Best Model — XGBoost Shuffled)

| AQI Category                 | Range   | Test MAE |
| ---------------------------- | ------- | -------- |
| Good                         | 0–50    | 12.3     |
| Moderate                     | 51–100  | 28.7     |
| Unhealthy (Sensitive Groups) | 101–150 | 42.1     |
| Unhealthy                    | 151–200 | 58.4     |
| Very Unhealthy               | 201–300 | 76.8     |
| Hazardous                    | >300    | 112.5    |

### Top Insights

- **PM2.5** (raw, rolling average, and city average) accounts for **40%+** of total feature importance
- **Temperature** and **humidity** rank in the top 10 predictors, reflecting their role in pollutant dispersion
- **Seasonal/monthly** features capture smog episodes driven by crop burning (Oct–Nov) and winter inversions (Dec–Jan)
- Model errors increase monotonically with AQI severity, largely due to class imbalance (hazardous days ≈ 3.2% of data)

---

## 🗂️ Dataset

| Property                | Value                      |
| ----------------------- | -------------------------- |
| Observations            | 18,247 city-date records   |
| Cities                  | 10                         |
| Temporal span           | Jan 2020 – Nov 2025        |
| Raw engineered features | 64                         |
| Final selected features | 27                         |
| Target                  | Multi-day AQI (regression) |

**Sources:** IQAir · OpenAQ API · Punjab/Sindh/KPK/Balochistan EPA dashboards · Pakistan Meteorological Department · OpenWeatherMap · NASA POWER · Kaggle archives

**Final Feature Set:**

| Category                    | Features                                                       |
| --------------------------- | -------------------------------------------------------------- |
| Pollutants (raw)            | PM2.5, NO, CO, SO2, O3                                         |
| Pollutants (lag)            | CO_lag1                                                        |
| Pollutants (rolling, 3-day) | PM2.5_roll3, NO_roll3, CO_roll3, SO2_roll3, O3_roll3           |
| Pollutants (city average)   | PM2.5_cityavg, NO_cityavg, CO_cityavg, SO2_cityavg, O3_cityavg |
| Meteorology                 | Temperature, Humidity, Wind Speed, Rainfall                    |
| Temporal                    | Day, Month, Weekday, Season_enc                                |
| Spatial                     | City_enc                                                       |

---

## 🛠️ Tech Stack

<div align="center">

**Language**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)

**ML / Modeling**

![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-006400?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-9ACD32?style=flat-square)

**Data Processing**

![Pandas](https://img.shields.io/badge/Pandas-2.1.0-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24.0-013243?style=flat-square&logo=numpy&logoColor=white)

**Visualization**

![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.0-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.0-4C72B0?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-5.17.0-3F4F75?style=flat-square&logo=plotly&logoColor=white)

**Web App**

![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

**Model Serialization**

![joblib](https://img.shields.io/badge/joblib-Model%20Serialization-FFA500?style=flat-square)

**Scraping / Ingestion**

![Requests](https://img.shields.io/badge/Requests-HTTP%20Library-000000?style=flat-square)
![BeautifulSoup4](https://img.shields.io/badge/BeautifulSoup4-Web%20Scraping-4B8BBE?style=flat-square)
![Selenium](https://img.shields.io/badge/Selenium-WebDriver-43B02A?style=flat-square&logo=selenium&logoColor=white)

**Training Environment**

![Google Colab](https://img.shields.io/badge/Google%20Colab-Tesla%20T4%20GPU-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

</div>

| Layer                    | Technologies                                      |
| ------------------------ | ------------------------------------------------- |
| **Language**             | Python 3.10                                       |
| **ML / Modeling**        | scikit-learn 1.3.0, XGBoost 2.0.0, LightGBM 4.1.0 |
| **Data Processing**      | pandas 2.1.0, numpy 1.24.0                        |
| **Visualization**        | matplotlib 3.7.0, seaborn 0.12.0, Plotly 5.17.0   |
| **Web App**              | Streamlit 1.28.0                                  |
| **Model Serialization**  | joblib                                            |
| **Scraping / Ingestion** | requests, BeautifulSoup4, Selenium WebDriver      |
| **Training Environment** | Google Colab (Tesla T4 GPU)                       |

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/username/ecocast-intelligence.git
cd ecocast-intelligence

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt` (core dependencies)

```
streamlit==1.28.0
plotly==5.17.0
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.1.0
pandas==2.1.0
numpy==1.24.0
matplotlib==3.7.0
seaborn==0.12.0
joblib
beautifulsoup4
selenium
requests
```

---

## ▶️ Usage

### Run the Web Application Locally

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Using the App

1. **Select a city** from the dropdown (10 supported cities)
2. **Choose a date** for next-day AQI prediction
3. **Input pollutant & weather values** (or click _"Use Historical Average"_ for auto-fill)
4. Click **Predict AQI** to view:
   - 🔢 Numerical AQI prediction with confidence interval
   - 🎨 Color-coded AQI category (Good → Hazardous)
   - 🏥 Health risk advisory tailored to sensitive populations
   - 📈 Historical trends, city comparisons, and feature importance charts

### Retrain the Model

```bash
python train_model.py --data data/processed/aqi_dataset.csv --output models/xgb_aqi_model.pkl
```

---

## 🧠 Methodology Summary

1. **Data Integration** — Harmonized 4 heterogeneous sources to daily resolution; unit standardization (µg/m³, ppm); EPA sub-index AQI recalculation
2. **Cleaning** — Tiered imputation (city-wise mean → temporal interpolation → global fallback); IQR-based outlier treatment
3. **Feature Engineering** — 64 candidate features reduced to 27 via correlation filtering and data-leakage removal
4. **Modeling** — Temporal (non-shuffled) 80/20 train-test split to prevent leakage; 7 models benchmarked
5. **Evaluation** — MAE, RMSE, R² on held-out future data; per-AQI-category error breakdown

---
