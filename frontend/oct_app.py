from PIL import Image, ImageOps
import streamlit as st
import requests


st.title("Retinal OCT Classification")

uploaded_file = st.file_uploader("Please Choose an OCT Image...", type=([".jpg", ".jpeg", ".png"]))
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    files = {"file": uploaded_file.getvalue()}
    st.image(image, caption='Uploaded OCT Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    #pred = requests.post(f"http://localhost:8000/predict", files = files) # connects to backend/predict url.
    pred = requests.post(f"http://backend:8000/predict", files = files) # connects to backend/predict url.


    pred_path = pred.json()
    prediction = pred_path.get("Prediction") # gets the prediction class from the json file

    if prediction == 'CNV':
        st.write("The OCT Image is classified as CNV")
    if prediction == "DME":
        st.write("The OCT Image is classified as DME")
    if prediction == "DRUSEN":
        st.write("The OCT Image is classified as DRUSEN")
    elif prediction == "NORMAL":
        st.write("The OCT Image is NORMAL")
