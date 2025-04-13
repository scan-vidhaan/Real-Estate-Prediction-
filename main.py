import joblib
import pandas as pd
import numpy as np
import openai
import streamlit as st

# 🔐 Set OpenAI API key
openai.api_key = ""  # Replace with your actual key securely

# Load models
model_options = {
    "Decision Tree (Base)": joblib.load("dt_model_base.pkl"),
    "Random Forest (Base)": joblib.load("random_forest_mode_base.pkl"),
    "XGBoost (Base)": joblib.load("xgb_mode_base.pkl"),
    "Decision Tree (Tuned)": joblib.load("dttuned_model.pkl"),
    "Random Forest (Tuned)": joblib.load("rftuned_model.pkl"),
    "XGBoost (Tuned)": joblib.load("xgbtuned_model.pkl"),
}

# Load feature columns
feature_columns = joblib.load("feature_columns.pkl")

# Extract categorical options
location_cols = [col for col in feature_columns if col.startswith("location_")]
locations = [col.replace("location_", "") for col in location_cols]

area_cols = [col for col in feature_columns if col.startswith("area_type_")]
area_types = [col.replace("area_type_", "") for col in area_cols]

# Load raw dataset for calculating psf dynamically
raw_df = pd.read_csv("Bengaluru_House_Data.csv")
raw_df["total_sqft"] = pd.to_numeric(raw_df["total_sqft"], errors='coerce')
raw_df = raw_df.dropna(subset=["total_sqft", "price"])
raw_df["psf"] = (raw_df["price"] * 100000) / raw_df["total_sqft"]

# UI setup
st.title("🏘️ Bangalore Property Price Predictor")
st.markdown("Enter property details to get the **predicted price (₹ Lakhs)**.")

# Model selection
model_choice = st.selectbox("Select a Model", list(model_options.keys()))
model = model_options[model_choice]

# User Inputs - Numerical
availability = st.text_input("Availability", "Ready To Move")
size = st.number_input("BHK (Size)", min_value=1, max_value=10, value=2)
total_sqft = st.number_input("Total Square Feet", min_value=200, max_value=10000, value=1000)
bath = st.slider("Number of Bathrooms", 1, 5, 2)
balcony = st.slider("Number of Balconies", 0, 3, 1)

# Categorical inputs
area_type = st.selectbox("Area Type", area_types)
location = st.selectbox("Location", locations)

# Calculate location-based psf dynamically
loc_df = raw_df[raw_df["location"].str.lower() == location.lower()]

# Group by location and calculate the mean for numerical columns
location_grouped = loc_df.groupby("location").agg(
    median_psf=("psf", "median"),
    low_psf=("psf", lambda x: x.quantile(0.33)),
    high_psf=("psf", lambda x: x.quantile(0.66)),
).reset_index()

# Check if the location exists in the grouped data
location_data = location_grouped[location_grouped["location"].str.lower() == location.lower()]

if location_data.empty:
    st.error(f"Location '{location}' is not available in the dataset.")
else:
    location_data = location_data.iloc[0]  # Safely select the first row

    # Round the median PSF to the nearest integer
    rounded_median_psf = round(location_data['median_psf'])

    # Calculate PSF dynamically
    if total_sqft > 0:
        actual_psf = (size * total_sqft) / total_sqft
        # Compare actual PSF with rounded median PSF
        if actual_psf <= rounded_median_psf:
            psf_category = 0  # Low PSF
        elif actual_psf <= (rounded_median_psf * 1.25):  # For medium category, let's use 25% buffer
            psf_category = 1  # Medium PSF
        else:
            psf_category = 2  # High PSF
    else:
        psf_category = 1  # Default to Medium PSF if no area is given

    # Calculate PSF Difference from the median
    psf_diff = actual_psf - rounded_median_psf

    # Display location-specific details
    st.markdown(f"📍 Median PSF for {location}: ₹ {rounded_median_psf}")
    st.markdown(f"📊 PSF Difference: ₹ {psf_diff}")
    st.markdown(f"🧩 PSF Category: **{['Low', 'Medium', 'High'][psf_category]}**")

    # Preparing the input data for prediction
    input_data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    # Fill basic numerical values
    input_data.at[0, "availability"] = 1  # Assuming 'Ready To Move' is encoded as 1
    input_data.at[0, "size"] = size
    input_data.at[0, "total_sqft"] = total_sqft
    input_data.at[0, "bath"] = bath
    input_data.at[0, "balcony"] = balcony

    # One-hot encode area_type and location
    area_col = f"area_type_{area_type}"
    loc_col = f"location_{location}"
    if area_col in input_data.columns:
        input_data.at[0, area_col] = 1
    if loc_col in input_data.columns:
        input_data.at[0, loc_col] = 1

    # Set the PSF category as an additional feature (assuming it's part of the feature_columns)
    input_data.at[0, "psf_category"] = psf_category

    # Prediction
    if st.button("Predict Price"):
        try:
            predicted_price = model.predict(input_data)[0]
            st.success(f"💰 Predicted Property Price: ₹ {round(predicted_price, 2)} Lakhs")

            # OpenAI market forecast
            with st.spinner("Checking with market intelligence..."):
                prompt = (
                    f"A {size} BHK property in {location}, Bangalore with {bath} bathrooms, {balcony} balconies, "
                    f"{total_sqft} sq ft area, predicted at ₹ {round(predicted_price, 2)} lakhs. "
                    f"Is this price realistic for Bangalore in 2025? "
                    f"Also, forecast its value in 5 years if the market trend continues."
                )
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=3000
                )
                ai_response = response['choices'][0]['message']['content']
                st.markdown("### 🤖 Market Assessment & Future Price Estimate")
                st.info(ai_response)

        except Exception as e:
            st.error(f"Prediction or OpenAI API Error: {e}")
