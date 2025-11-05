import streamlit as st
import pandas as pd
import joblib

# Load model and column info
data = joblib.load("model.pkl")

model = data["model"]
model_columns = data["columns"]

st.title("✈️ Airline Fare Prediction App")
st.write("Predict flight ticket prices based on airline and journey details.")

# --- Input Fields ---

# Dropdowns (you can adjust options based on your dataset)
airline = st.selectbox("Airline", [
    'Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers', 'SpiceJet', 
    'Vistara', 'GoAir', 'Multiple carriers Premium economy', 'Jet Airways Business',
    'Vistara Premium economy', 'Trujet'
])

source = st.selectbox("Source", ['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Banglore'])
destination = st.selectbox("Destination", ['Cochin', 'Banglore', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata'])

total_stops = st.selectbox("Total Stops", ['non-stop', '1 stop', '2 stops', '3 stops', '4 stops'])

dep_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23, value=10)
dep_min = st.number_input("Departure Minute (0-59)", min_value=0, max_value=59, value=30)
arrival_hour = st.number_input("Arrival Hour (0-23)", min_value=0, max_value=23, value=14)
arrival_min = st.number_input("Arrival Minute (0-59)", min_value=0, max_value=59, value=45)
duration_hours = st.number_input("Duration (hours)", min_value=0.0, value=2.0)
duration_mins = st.number_input("Duration (minutes)", min_value=0.0, value=30.0)
days_left = st.number_input("Days Left for Travel", min_value=0, value=30)

# --- Create a dataframe for input ---
input_dict = {
    "Airline": [airline],
    "Source": [source],
    "Destination": [destination],
    "Total_Stops": [total_stops],
    "Dep_Time_hour": [dep_hour],
    "Dep_Time_minute": [dep_min],
    "Arrival_Time_hour": [arrival_hour],
    "Arrival_Time_minute": [arrival_min],
    "Duration_hours": [duration_hours],
    "Duration_mins": [duration_mins],
    "Days_Left": [days_left],
}

input_df = pd.DataFrame(input_dict)

# --- Match one-hot encoded columns ---
# Create dummy variables to match training data
input_encoded = pd.get_dummies(input_df)

# Align columns to training set (add missing ones with 0)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# --- Predict ---
if st.button("Predict Fare"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"Estimated Flight Fare: ₹{prediction:,.2f}")

