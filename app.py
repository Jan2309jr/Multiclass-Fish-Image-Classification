import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
# Streamlit page config
st.set_page_config(page_title="Fish Classifier", layout="centered")
# Load class names from classes.txt
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_mobilenet_model.h5", compile=False)
model = load_model()
st.title("🐟 Fish Image Classifier")
st.write("Upload a fish image to get its predicted category and confidence scores.")
# File uploader
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
    # Prediction
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100
    # Display main prediction
    st.markdown("### 🐠 Prediction")
    st.success(f"**Fish Type:** {predicted_label}")
    st.info(f"**Confidence:** {confidence:.2f}%")
    # Create DataFrame for all class probabilities
    df_probs = pd.DataFrame({
        "Class": class_names,
        "Confidence (%)": predictions * 100
    }).sort_values(by="Confidence (%)", ascending=False)
    # Show as bar chart
    st.markdown("### 📊 Confidence Scores")
    st.bar_chart(df_probs.set_index("Class"))
    st.dataframe(df_probs.reset_index(drop=True))