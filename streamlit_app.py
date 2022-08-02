from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
from api.image_augmentation import RandomColorAffine
import streamlit as st

st.title("Retinal OCT Classification")

def oct_classification(img):

    #load the model
    model = load_model("models/10%_simclr_semi_supervised_model_bs64.33-0.27_080222.h5",custom_objects={'RandomColorAffine': RandomColorAffine})    

    #turn the image into a numpy array and resize
    size = (224, 224)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_resized = tf.keras.preprocessing.image.smart_resize(img_array, size)

    # Load the image into a batch of 1
    img_batch = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img_batch[0] = img_resized

    # run the inference
    prediction = model.predict(img_batch)
    return np.argmax(prediction[0])

uploaded_file = st.file_uploader("Please Choose an OCT Image...", type=([".jpg", ".jpeg", ".png"]))
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded OCT Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = oct_classification(image)
    if label == 0:
        st.write("The OCT Image is classified as CNV")
    if label == 1:
        st.write("The OCT Image is classified as DME")
    if label == 2:
        st.write("The OCT Image is classified as DRUSEN")
    elif label == 3:
        st.write("The OCT Image is NORMAL")
