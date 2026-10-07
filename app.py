import streamlit as st
import pickle 
import pandas as pd
st.set_page_config( page_title="Customer Segmentation", page_icon="👥", layout="centered" )
st.title("👥 Customer Segmentation")
st.write("Enter customer information to predict the customer segment.")
with open("customer_segmentation_model.pkl", "rb") as file:
    model = pickle.load(file)
with open("customer_segmentation_scaler.pkl", "rb") as file:
    scalling = pickle.load(file)
recency = st.number_input( "Recency", min_value=0.0, value=30.0 )
Total_Spending = st.number_input( "Total Spending", min_value=0.0, value=1000.0 )
Purchase_Count = st.number_input( "Purchase Count", min_value=0.0, value=10.0 )
if st.button("Predict Segment"):
    feature_select = pd.DataFrame( [[recency, Total_Spending, Purchase_Count]], columns=["recency", "Total_Spending", "Purchase_Count"] )
    fit_transform = scalling.transform(feature_select)
    prediction = model.predict(fit_transform)
    cluster_mapping = { 0: "HIGH", 1: "MEDIUM", 2: "LOW" }
    segment = cluster_mapping[prediction[0]]
    st.success(f"Customer Segment: {segment}")