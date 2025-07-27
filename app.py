import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data and pre-process
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("CarPricePrediction.csv")
    # Create age from year
    df['age'] = 2025 - df['year']
    df.drop(['year','name'], axis=1, inplace=True)

    # Encode categorical columns
    categorical_cols = ['fuel','seller_type','transmission','owner']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

df = load_and_process_data()

# Split features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a linear regression model
model = LinearRegression()
model.fit(X_scaled, y)

st.title("Car Price Predictor")

st.markdown("""
Use the sidebar to input car features, then click 'Predict Price' to get the estimated selling price.
""")

# Sidebar inputs
st.sidebar.header("Car Features Input")

# Age input (derived from manufacture year)
age = st.sidebar.number_input("Car Age (years)", min_value=0, max_value=50, value=5)

km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000)

# Categorical inputs for original columns that were one-hot encoded
# We'll create one-hot encoded vectors based on these inputs

fuel_options = ['Petrol', 'Diesel', 'CNG', 'LPG']  # from data
fuel = st.sidebar.selectbox("Fuel Type", fuel_options)

seller_options = df.filter(regex='seller_type_').columns.str.replace('seller_type_', '').tolist()
# Add back the dropped first category 'Individual' if missing in columns
if 'Individual' not in seller_options: 
    seller_options = ['Individual'] + seller_options
seller_type = st.sidebar.selectbox("Seller Type", seller_options)

transmission_options = df.filter(regex='transmission_').columns.str.replace('transmission_', '').tolist()
if 'Manual' not in transmission_options:
    transmission_options = ['Manual'] + transmission_options
transmission = st.sidebar.selectbox("Transmission", transmission_options)

owner_options = df.filter(regex='owner_').columns.str.replace('owner_', '').tolist()
if 'First Owner' not in owner_options:
    owner_options = ['First Owner'] + owner_options
owner = st.sidebar.selectbox("Owner Type", owner_options)

# Prepare input vector for prediction
def encode_input(age, km_driven, fuel, seller_type, transmission, owner):
    # Start with zeros for all features except age and km_driven
    input_dict = {col: 0 for col in X.columns}

    # Numerical features
    input_dict['age'] = age
    input_dict['km_driven'] = km_driven

    # Encoding fuel
    fuel_col = f'fuel_{fuel}'
    if fuel_col in input_dict:
        input_dict[fuel_col] = 1
    # If fuel is the dropped first category, no column set to 1

    # Encoding seller_type
    st_col = f'seller_type_{seller_type}'
    if st_col in input_dict:
        input_dict[st_col] = 1

    # Encoding transmission
    tr_col = f'transmission_{transmission}'
    if tr_col in input_dict:
        input_dict[tr_col] = 1

    # Encoding owner
    own_col = f'owner_{owner}'
    if own_col in input_dict:
        input_dict[own_col] = 1

    return pd.DataFrame([input_dict])

input_df = encode_input(age, km_driven, fuel, seller_type, transmission, owner)

# Scale input
input_scaled = scaler.transform(input_df)

if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    st.success(f"Estimated Selling Price: ₹ {prediction[0]:,.0f}")

st.write("---")
st.header("Sample Data Preview")
st.dataframe(df.head())
