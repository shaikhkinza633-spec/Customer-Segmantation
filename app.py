import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="centered"
)

st.title("👥 Customer Segmentation")
st.write("Enter customer information to predict the customer segment.")

with open("model_1.pkl", "rb") as file:
    model = pickle.load(file)
with open("scalling_1.pkl", "rb") as file:
    scalling = pickle.load(file)


cluster_mapping = {
    0: "HIGH",
    1: "MEDIUM",
    2: "LOW"
}


recency = st.number_input(
    "Recency (days since last purchase)",
    min_value=0,
    value=30
)

total_spending = st.number_input(
    "Total Spending",
    min_value=0.0,
    value=1500.0
)

purchase_count = st.number_input(
    "Purchase Count",
    min_value=0,
    value=20
)

if st.button("Predict Segment"):
    new_customer = pd.DataFrame({
        "recency": [recency],
        "Total_Spending": [total_spending],
        "Purchase_Count": [purchase_count]
    })

    scaled_data = scalling.transform(new_customer)
    prediction = int(model.predict(scaled_data)[0])
    segment = cluster_mapping[prediction]

    st.success(f"Customer Segment: {segment}")
