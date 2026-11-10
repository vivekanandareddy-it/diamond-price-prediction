import streamlit as st
import pandas as pd
import joblib

model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💎 Diamond Price Prediction")

carat = st.number_input("Carat", min_value=0.1, value=0.5)

clarity = st.selectbox(
    "Clarity",
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
)

depth = st.number_input("Depth", value=61.5)
table = st.number_input("Table", value=57.0)
y = st.number_input("Width (y)", value=4.25)

if st.button("Predict Price"):

    clarity_map = {
        "I1": 0,
        "SI2": 1,
        "SI1": 2,
        "VS2": 3,
        "VS1": 4,
        "VVS2": 5,
        "VVS1": 6,
        "IF": 7
    }

    clarity_encoded = clarity_map[clarity]

    input_data = pd.DataFrame({
        "carat": [carat],
        "depth": [depth],
        "table": [table],
        "y": [y]
    })

    input_scaled = scaler.transform(input_data)

    input_variable = pd.DataFrame({
        "carat": [input_scaled[0, 0]],
        "clarity": [clarity_encoded],
        "depth": [input_scaled[0, 1]],
        "table": [input_scaled[0, 2]],
        "y": [input_scaled[0, 3]]
    })

    prediction = model.predict(input_variable)

    st.success(f"Predicted Diamond Price: ${prediction[0]:,.2f}")