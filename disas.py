# Save this as flood_dashboard.py and run with: streamlit run flood_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ----------------------
# 1. Prepare Sample Data & Model
# ----------------------
data = pd.DataFrame({
    'rainfall_mm': [10, 50, 120, 5, 200, 80, 0, 300],
    'river_level_m': [1, 3, 5, 0.5, 6, 4, 0.2, 7],
    'humidity_percent': [60, 80, 90, 50, 95, 85, 40, 98],
    'flood_risk': ['Low', 'Medium', 'High', 'Low', 'High', 'Medium', 'Low', 'High']
})

X = data[['rainfall_mm', 'river_level_m', 'humidity_percent']]
y = data['flood_risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ----------------------
# 2. Streamlit Dashboard
# ----------------------
st.title("🌊 Flood Forecasting & Alert System")
st.write("Enter current weather and river data to predict flood risk:")

# User Inputs
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=50.0)
river_level = st.number_input("River Level (m)", min_value=0.0, value=2.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)

# Prediction Button
if st.button("Predict Flood Risk"):
    X_input = np.array([[rainfall, river_level, humidity]])
    X_input_scaled = scaler.transform(X_input)
    risk = model.predict(X_input_scaled)[0]
    
    # Display Alert
    if risk == 'High':
        st.error(f"⚠️ HIGH flood risk detected! Take precautions immediately!")
    elif risk == 'Medium':
        st.warning(f"ℹ️ Medium flood risk. Stay alert and monitor updates.")
    else:
        st.success(f"✅ Low flood risk. All clear.")

# Optional: Display sample dataset
st.subheader("Sample Dataset Used for Model")
st.dataframe(data)
