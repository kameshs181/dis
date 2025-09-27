# flood_lstm_dashboard.py

# ----------------------
# Imports
# ----------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import folium
from streamlit_folium import st_folium

# ----------------------
# 1. Load Dataset
# ----------------------
st.title("🌊 Advanced Flood Forecasting Dashboard (LSTM)")

# Load sample CSV
data = pd.read_csv("sample_flood_data.csv", parse_dates=['date'])

st.subheader("Sample Flood Data")
st.dataframe(data.head())

# Fill missing values
data.fillna(method='ffill', inplace=True)

# ----------------------
# 2. Feature Engineering
# ----------------------
data['rainfall_intensity'] = data['rainfall_mm']
data['river_change'] = data['river_level_m'].diff().fillna(0)
data['month'] = data['date'].dt.month
data['week'] = data['date'].dt.isocalendar().week

# Select features for LSTM
features = ['rainfall_mm', 'humidity_percent', 'river_level_m']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# ----------------------
# 3. Create Sequences
# ----------------------
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 2])  # river_level_m
    return np.array(X), np.array(y)

seq_length = 5
X_seq, y_seq = create_sequences(scaled_data, seq_length)

st.write(f"Number of sequences for LSTM: {X_seq.shape[0]}")

# Only proceed if we have enough sequences
if X_seq.shape[0] > 0:
    # ----------------------
    # 4. LSTM Model
    # ----------------------
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2])))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    
    # Train (few epochs for demo)
    lstm_model.fit(X_seq, y_seq, epochs=10, batch_size=8, verbose=0)
    
    # Predict next day river level
    last_seq = scaled_data[-seq_length:].reshape(1, seq_length, len(features))
    pred_scaled = lstm_model.predict(last_seq)
    pred_river_level = scaler.inverse_transform(np.array([[0,0,pred_scaled[0][0]]]))[0][2]
    
    st.subheader("Predicted River Level (Next Day)")
    st.metric(label="Predicted River Level (m)", value=round(pred_river_level,2))
    
    # ----------------------
    # 5. Risk Classification
    # ----------------------
    if pred_river_level < 2:
        risk_level = "Low"
    elif 2 <= pred_river_level < 5:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    st.subheader("Flood Risk Level")
    if risk_level == "High":
        st.error("⚠️ HIGH FLOOD RISK ALERT!")
    elif risk_level == "Medium":
        st.warning("ℹ️ Medium flood risk. Stay alert.")
    else:
        st.success("✅ Low flood risk. All clear.")
    
    # ----------------------
    # 6. Map Visualization
    # ----------------------
    st.subheader("Flood Risk Map")
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=7)
    
    for _, row in data.iterrows():
        if risk_level == "High":
            color = 'red'
        elif risk_level == "Medium":
            color = 'orange'
        else:
            color = 'green'
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    
    st_folium(m, width=700)
else:
    st.warning("Not enough data to train LSTM. Add more rows or reduce sequence length.")
