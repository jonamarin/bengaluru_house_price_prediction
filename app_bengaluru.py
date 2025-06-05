import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("bengaluru_house_prices.csv")
    df = df.copy()

    # Drop rows with missing price or sqft
    df = df.dropna(subset=['total_sqft', 'price'])

    # Convert total_sqft to numeric, handle non-numeric cases
    def convert_sqft(s):
        try:
            return float(s)
        except:
            if '-' in str(s):
                parts = s.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            return np.nan

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    df = df.dropna(subset=['total_sqft'])

    # Extract number of bedrooms
    df['bhk'] = df['size'].str.extract(r'(\d+)').astype(float)

    # Fill missing values
    df['bath'] = df['bath'].fillna(df['bath'].median())
    df['balcony'] = df['balcony'].fillna(df['balcony'].median())
    df['bhk'] = df['bhk'].fillna(df['bhk'].median())

    # Label encode categorical features
    le_area = LabelEncoder()
    le_loc = LabelEncoder()

    df['area_type'] = le_area.fit_transform(df['area_type'].astype(str))
    df['location'] = le_loc.fit_transform(df['location'].astype(str))

    # Final feature set
    features = ['area_type', 'location', 'total_sqft', 'bath', 'balcony', 'bhk']
    X = df[features]
    y = df['price']

    return X, y, le_area, le_loc

# Load data
X, y, le_area, le_loc = load_data()

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("🏠 House Price Predictor")

st.write("Enter the property details below to estimate the price (in Lakhs ₹):")

area_type_input = st.selectbox("Area Type", le_area.classes_)
location_input = st.selectbox("Location", le_loc.classes_)
total_sqft_input = st.number_input("Total Square Feet", min_value=100.0, max_value=10000.0, step=50.0)
bath_input = st.slider("Number of Bathrooms", 1, 10, 2)
balcony_input = st.slider("Number of Balconies", 0, 5, 1)
bhk_input = st.slider("Number of Bedrooms (BHK)", 1, 10, 2)

# Prepare input
if st.button("Predict Price"):
    input_data = pd.DataFrame([[
        le_area.transform([area_type_input])[0],
        le_loc.transform([location_input])[0],
        total_sqft_input,
        bath_input,
        balcony_input,
        bhk_input
    ]], columns=['area_type', 'location', 'total_sqft', 'bath', 'balcony', 'bhk'])

    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")

