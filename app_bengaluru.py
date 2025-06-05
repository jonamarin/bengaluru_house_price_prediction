import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    df['bhk'] = df['size'].str.extract(r'(\d+)').astype(float)
    df['bath'] = df['bath'].fillna(df['bath'].median())
    df['balcony'] = df['balcony'].fillna(df['balcony'].median())
    df['bhk'] = df['bhk'].fillna(df['bhk'].median())

    le_area = LabelEncoder()
    le_loc = LabelEncoder()

    df['area_type_encoded'] = le_area.fit_transform(df['area_type'].astype(str))
    df['location_encoded'] = le_loc.fit_transform(df['location'].astype(str))

    # Final feature set
    features = ['area_type_encoded', 'location_encoded', 'total_sqft', 'bath', 'balcony', 'bhk']
    X = df[features]
    y = df['price']

    return df, X, y, le_area, le_loc

# Load data
df, X, y, le_area, le_loc = load_data()

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Tabs for navigation
tab1, tab2 = st.tabs(["🏠 Price Prediction", "📊 EDA Dashboard"])

# ---------------- TAB 1: Prediction ----------------
with tab1:
    st.title("🏠 Bengaluru Houses Price Predictor")

    st.write("Enter the property details below to estimate the price (in Lakhs ₹):")

    area_type_input = st.selectbox("Area Type", le_area.classes_)
    location_input = st.selectbox("Location", le_loc.classes_)
    total_sqft_input = st.number_input("Total Square Feet", min_value=100.0, max_value=10000.0, step=50.0)
    bath_input = st.slider("Number of Bathrooms", 1, 10, 2)
    balcony_input = st.slider("Number of Balconies", 0, 5, 1)
    bhk_input = st.slider("Number of Bedrooms (BHK)", 1, 10, 2)

    if st.button("Predict Price"):
        input_data = pd.DataFrame([[
            le_area.transform([area_type_input])[0],
            le_loc.transform([location_input])[0],
            total_sqft_input,
            bath_input,
            balcony_input,
            bhk_input
        ]], columns=['area_type_encoded', 'location_encoded', 'total_sqft', 'bath', 'balcony', 'bhk'])

        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")

# ---------------- TAB 2: EDA ----------------
with tab2:
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Distribution of Price")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['price'], bins=50, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Price vs Total Square Feet")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='total_sqft', y='price', data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Average Price by Location")
    top_locations = df['location'].value_counts().head(10).index
    loc_avg_price = df[df['location'].isin(top_locations)].groupby('location')['price'].mean().sort_values()
    fig3, ax3 = plt.subplots()
    loc_avg_price.plot(kind='barh', ax=ax3)
    ax3.set_xlabel("Average Price (Lakhs ₹)")
    st.pyplot(fig3)

    st.subheader("BHK vs Price Boxplot")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x='bhk', y='price', data=df, ax=ax4)
    st.pyplot(fig4)

    st.subheader("Correlation Heatmap")
    fig5, ax5 = plt.subplots()
    corr = df[['price', 'total_sqft', 'bath', 'balcony', 'bhk']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)
