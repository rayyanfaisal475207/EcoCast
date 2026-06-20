import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from feature_pipeline import FeaturePipeline
from datetime import datetime, timedelta

# ----------------- Configuration -----------------
st.set_page_config(
    page_title="EcoCast | Smart Air Analytics",
    page_icon="graph_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- Premium CSS System -----------------
st.markdown("""
<style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        color: #1A202C;
        background-color: #F7FAFC;
    }
    
    /* Hide Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header Section */
    .hero-container {
        text-align: center;
        padding: 40px 20px;
        background: white;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 40px;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #2D3748;
        letter-spacing: -1px;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #718096;
        margin-top: 10px;
    }
    
    /* Card/Box Styling */
    .content-box {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #EDF2F7;
        margin-bottom: 30px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .content-box:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Sections */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2D3748;
        margin-bottom: 24px;
        border-left: 5px solid #3182CE;
        padding-left: 15px;
    }
    
    /* Custom Slider Styling */
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
        background-color: #3182CE; 
        box-shadow: 0 0 0 5px rgba(49, 130, 206, 0.2);
    }
    div.stSlider > div[data-baseweb="slider"] > div > div > div > div {
        background-color: #63B3ED;
    }
    
    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(to right, #3182CE, #2B6CB0);
        color: white;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.4);
    }
    
    /* Stats Footer */
    .stat-card {
        text-align: center;
        padding: 20px;
        background: #EBF8FF;
        border-radius: 12px;
        color: #2C5282;
    }
    .stat-value { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #4299E1; }
    
</style>
""", unsafe_allow_html=True)

# ----------------- Load Data & Model -----------------
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('xgboost_model.pkl')
        pipeline = FeaturePipeline('city_metadata.json')
        return model, pipeline
    except Exception as e:
        return None, None

model, pipeline = load_resources()

if not model:
    st.error("System Error: Resources missing.")
    st.stop()

# ----------------- HERO SECTION -----------------
st.markdown("""
<div class="hero-container">
    <div style="font-size: 4rem;">🌍</div>
    <h1 class="hero-title">EcoCast Intelligence</h1>
    <p class="hero-subtitle">Advanced Air Quality Monitoring & Forecasting System</p>
</div>
""", unsafe_allow_html=True)

# ----------------- PREDICTION ENGINE -----------------
st.markdown('<div class="section-header">Live Prediction Engine</div>', unsafe_allow_html=True)

# Main Container
with st.container():
    # Top Control Bar (City & Date)
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        city_meta = pipeline.city_meta
        city_list = sorted([city_meta[k]['name'] for k in city_meta])
        sel_city = st.selectbox("Select Target City", city_list)
        sel_id = next(int(k) for k, v in city_meta.items() if v['name'] == sel_city)
    with c2:
        sel_date = st.date_input("Target Date", datetime.now())
    st.markdown('</div>', unsafe_allow_html=True)

    # Input Grid
    col_inputs, col_results = st.columns([1.2, 0.8], gap="large")

    with col_inputs:
        # POLLUTANTS
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("🏭 Emission Levels")
        pm25 = st.slider("Particulate Matter (PM2.5)", 0, 500, 85, help="Micrograms per cubic meter")
        no = st.slider("Nitrogen Oxide (NO)", 0, 300, 25)
        co = st.slider("Carbon Monoxide (CO)", 0.0, 50.0, 1.2)
        so2 = st.slider("Sulfur Dioxide (SO2)", 0, 200, 12)
        o3 = st.slider("Ozone (O3)", 0, 400, 45)
        st.markdown('</div>', unsafe_allow_html=True)

        # WEATHER
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("☁️ Meteorology")
        w1, w2 = st.columns(2)
        with w1:
            temp = st.number_input("Temperature (°C)", -10.0, 50.0, 28.0)
            wind = st.number_input("Wind Speed (m/s)", 0.0, 50.0, 4.0)
        with w2:
            hum = st.number_input("Humidity (%)", 0, 100, 60)
            rain = st.number_input("Rainfall (mm)", 0.0, 200.0, 0.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        # CALCULATION
        input_data = {
            'City_enc': sel_id, 'Date': pd.to_datetime(sel_date),
            'PM2.5': pm25, 'NO': no, 'CO': co, 'SO2': so2, 'O3': o3,
            'Temperature': temp, 'Humidity': hum, 'Wind_speed': wind, 'Rainfall': rain
        }
        features = pipeline.preprocess(input_data)
        pred = model.predict(features)[0]

        # Logic
        if pred <= 50: status, color = "Good", "#48BB78"
        elif pred <= 100: status, color = "Moderate", "#ECC94B"
        elif pred <= 150: status, color = "Sensitive", "#ED8936"
        elif pred <= 200: status, color = "Unhealthy", "#F56565"
        elif pred <= 300: status, color = "Very Unhealthy", "#9F7AEA"
        else: status, color = "Hazardous", "#2D3748"

        # DISPLAY CARD
        st.markdown(f"""
        <div class="content-box" style="text-align: center; border-top: 8px solid {color}; background: #FDFFFF;">
            <h3 style="color: #A0AEC0; margin-bottom: 10px;">AIR QUALITY INDEX</h3>
            <div style="font-size: 6rem; font-weight: 800; color: {color}; line-height: 1; margin-bottom: 10px;">
                {pred:.0f}
            </div>
            <div style="background: {color}; color: white; display: inline-block; padding: 10px 20px; border-radius: 50px; font-weight: 700; margin-bottom: 20px;">
                {status}
            </div>
            <hr style="border-top: 1px solid #E2E8F0;">
            <p style="color: #718096; font-size: 0.9rem;">
                Based on XGBoost Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # GAUGE
        st.markdown('<div class="content-box" style="padding: 10px;">', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = pred,
            gauge = {
                'axis': {'range': [0, 500]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "#C6F6D5"},
                    {'range': [50, 100], 'color': "#FEFCBF"},
                    {'range': [100, 150], 'color': "#FEEBC8"},
                    {'range': [150, 200], 'color': "#FED7D7"},
                    {'range': [200, 500], 'color': "#E9D8FD"}
                ]
            }
        ))
        fig_gauge.update_layout(height=250, margin={'t':20,'b':20,'l':20,'r':20}, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- ANALYTICS VISUALIZATION SECTION -----------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Environmental Insights</div>', unsafe_allow_html=True)

vis_col1, vis_col2 = st.columns([1.2, 0.8])

with vis_col1:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("📅 7-Day Trend Forecast")
    
    # Forecast Logic
    forecast_dates = [pd.to_datetime(sel_date) + timedelta(days=i) for i in range(8)]
    forecast_vals = []
    
    for d in forecast_dates:
        # Feature Simulation
        sim = input_data.copy()
        sim['Date'] = d
        sim['Temperature'] += np.sin(d.day) * 1.5 # Natural variation
        sim_feat = pipeline.preprocess(sim)
        forecast_vals.append(model.predict(sim_feat)[0])
        
    fig_line = px.area(
        x=forecast_dates, y=forecast_vals,
        labels={'x': 'Date', 'y': 'Predicted AQI'},
    )
    fill_color_rgba = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)" if color.startswith('#') and len(color) == 7 else "rgba(100,100,100,0.1)"
    fig_line.update_traces(line_color=color, line_width=3, fillcolor=fill_color_rgba)
    fig_line.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white", xaxis_title=None)
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with vis_col2:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("🧪 Pollutant Composition")
    
    # Donut Chart for Pollutants
    p_data = pd.DataFrame({
        'Pollutant': ['PM2.5', 'NO', 'CO', 'SO2', 'O3'],
        'Value': [pm25, no, co*10, so2, o3] # Weighted for visibility
    })
    
    fig_pie = px.pie(
        p_data, values='Value', names='Pollutant',
        hole=0.6,
        color_discrete_sequence=px.colors.sequential.Bluyl
    )
    fig_pie.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Stats Footer (Simplified)
st.markdown("""<div class="content-box" style="text-align: center; background: #2D3748; color: white;">
    <h3 style="color: white; margin-bottom: 20px;">System Diagnostics</h3>
    <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
        <div>
            <div style="font-size: 2rem; font-weight: bold;">10</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">CITIES MONITORED</div>
        </div>
        <div>
            <div style="font-size: 2rem; font-weight: bold;">{:.2f}</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">MODEL R² SCORE</div>
        </div>
        <div>
            <div style="font-size: 2rem; font-weight: bold;">LIVE</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">INFERENCE STATUS</div>
        </div>
    </div>
</div>""".format(pipeline.metadata.get('model_accuracy', {}).get('R2', 0)), unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #CBD5E0; padding: 20px;">
    EcoCast Intelligence Suite v4.1 &copy; 2025
</div>
""", unsafe_allow_html=True)