import streamlit as st
import os
import sys



sys.path.insert(0, r"C:\Projects\plant-disease-capstone\src")

import tempfile
from predict import predict_image

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Plant Disease Detection And Recommendation System")

uploaded_file = st.file_uploader("Please upload a leaf image of tomato plant...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image",width=400)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name


    if st.button("Predict"):
        disease, confidence, recommendation = predict_image(temp_file_path)
        clean_disease = disease.replace("disease: ", "").strip()
        if confidence < 0.98:
            st.error("Image is not a tomato leaf. Please upload a valid tomato leaf image.") 
        else:
            st.success(f"Disease: {clean_disease}")
            st.info(f"Confidence: {confidence:.2f}")
            st.warning(f"Recommendation: {recommendation}")