import streamlit as st
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.express as px

# Load the trained LightGBM model
model = joblib.load("lightgbm_crop_yield_model.pkl")

# Load label encoders
label_encoders = joblib.load("label_encoders.pkl")

# Define mappings based on encoding output
region_mapping = {3: "West", 2: "South", 1: "North", 0: "East"}
soil_mapping = {4: "Sandy", 1: "Clay", 2: "Loam", 5: "Silt", 3: "Peaty", 0: "Chalky"}
crop_mapping = {1: "Cotton", 3: "Rice", 0: "Barley", 4: "Soybean", 5: "Wheat", 2: "Maize"}
weather_mapping = {0: "Cloudy", 1: "Rainy", 2: "Sunny"}

# Reverse mappings for dropdown selection
region_options = {v: k for k, v in region_mapping.items()}
soil_options = {v: k for k, v in soil_mapping.items()}
crop_options = {v: k for k, v in crop_mapping.items()}
weather_options = {v: k for k, v in weather_mapping.items()}

# Title and Description
st.title("🌾 Crop Yield Prediction for Farmers")
st.markdown("""
This tool helps farmers predict their crop yield based on key factors like:
- 🌡️ **Temperature**
- 🌧️ **Rainfall**
- 🏜️ **Soil Type**
- 🏞️ **Region**
Simply provide the details below, and we’ll give you an estimate!
""")

# Input Form
st.header("📋 Enter Your Farm Details")

# Use correct names but store numerical values
region = st.selectbox("🏞️ Select Your Region:", list(region_options.keys()))
soil_type = st.selectbox("🏜️ Select Soil Type:", list(soil_options.keys()))
crop = st.selectbox("🌾 Select Crop Type:", list(crop_options.keys()))
weather = st.selectbox("☁️ Select Weather Condition:", list(weather_options.keys()))

avg_temp = st.number_input("🌡️ Average Temperature (°C):", min_value=-10.0, max_value=50.0, step=0.1)
avg_rainfall = st.number_input("🌧️ Average Rainfall (mm):", min_value=0.0, step=0.1)
days_to_harvest = st.number_input("⏳ Days to Harvest:", min_value=30, max_value=365, step=1)

fertilizer_used = st.checkbox("🧪 Fertilizer Used? (Check if Yes)")
irrigation_used = st.checkbox("🚰 Irrigation Used? (Check if Yes)")

# Predict Yield Button
if st.button("🔍 Predict Yield"):
    # Convert categorical inputs into encoded values
    region_encoded = region_options[region]
    soil_encoded = soil_options[soil_type]
    crop_encoded = crop_options[crop]
    weather_encoded = weather_options[weather]

    # Convert boolean values to 0/1
    fertilizer_value = int(fertilizer_used)
    irrigation_value = int(irrigation_used)

    # Create input array (EXACT 9 FEATURES TO MATCH MODEL)
    input_data = np.array([[region_encoded, soil_encoded, crop_encoded, avg_rainfall, avg_temp,
                            fertilizer_value, irrigation_value, weather_encoded, days_to_harvest]])

    # Predict using the LightGBM model
    prediction = model.predict(input_data)[0]

    # Display Results
    st.subheader("📈 Prediction Result:")
    st.success(f"🌟 Your estimated crop yield is **{prediction:.2f} tons per hectare**.")

    # --- Add a Simple Bar Chart for Visualization ---
    st.subheader("📊 Yield Comparison")
    
    # Simulated average yield for comparison (can be replaced with real data)
    average_yield = 4.5  # Adjust based on real data

    yield_data = pd.DataFrame({
        "Category": ["Predicted Yield", "Average Yield"],
        "Yield (tons/hectare)": [prediction, average_yield]
    })

    fig = px.bar(yield_data, x="Category", y="Yield (tons/hectare)", color="Category",
                 title="Predicted vs. Average Crop Yield",
                 labels={"Yield (tons/hectare)": "Yield (tons per hectare)"},
                 text_auto=True)

    st.plotly_chart(fig)  # Display chart in Streamlit

    # --- Recommendations Based on Yield ---
    st.subheader("🌿 Farming Recommendations")
    
    if prediction < 3:
        st.warning("⚠️ **Low yield detected!** Consider improving soil quality and irrigation.")
        st.markdown("""
        - **Use organic compost** to enrich soil nutrients.  
        - **Improve irrigation** by ensuring crops get enough water.  
        - **Monitor soil pH** and adjust with lime or sulfur if needed.  
        """)
    elif prediction < 6:
        st.info("ℹ️ **Moderate yield detected.** You may optimize fertilizer and water usage.")
        st.markdown("""
        - **Increase nitrogen-based fertilizers** (e.g., urea) for better crop growth.  
        - **Adjust irrigation timing** based on soil moisture levels.  
        - **Ensure balanced crop rotation** to maintain soil fertility.  
        """)
    else:
        st.success("✅ **High yield expected!** Keep up the good farming practices!")
        st.markdown("""
        - **Maintain current farming techniques.**  
        - **Keep using organic or chemical fertilizers** as needed.  
        - **Monitor seasonal variations** to sustain high yield production.  
        """)

# Footer
st.write("---")
st.write("👨‍🌾 Developed to support farmers in making informed decisions! 🌱")