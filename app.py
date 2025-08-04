import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_mobilenet_model.h5", compile=False)

model = load_model()

# Define class names (update according to your dataset)
class_names = [
    "Betta", "Goldfish", "Guppy", "Angelfish", "Molly", 
    "Oscar", "Tetra", "Platy", "Swordtail", "Cichlid"
]

# Set title
st.set_page_config(page_title="Fish Classifier", layout="centered")
st.title("🐟 Fish Image Classifier")
st.write("Upload a fish image and get its predicted category and confidence score.")

# Upload image
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))  # MobileNet default size
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

    # Prediction
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    # Display result
    st.markdown("### 🐠 Prediction")
    st.success(f"**Fish Type:** {predicted_label}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Optional: Show all class probabilities
    st.markdown("### 📊 Confidence Scores")
    for i, prob in enumerate(predictions):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")
