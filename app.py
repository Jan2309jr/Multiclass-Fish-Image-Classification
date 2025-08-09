import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
from bs4 import BeautifulSoup

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_mobilenet_model", compile=False)  # folder name, not .h5

model = load_model()

# Define class names (update according to your dataset)
class_names = [
    "fish sea_food shrimp",
    "fish sea_food trout",
    "fish sea_food red_sea_bream",
    "animal fish",
    "fish sea_food gilt_head_bream",
    "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food striped_red_mullet",
    "fish sea_food black_sea_sprat",
    "fish sea_food sea_bass",
    "animal fish bass"
]

# Set Streamlit page config
st.set_page_config(page_title="Fish Classifier", layout="centered")
st.title("🐟 Fish Image Classifier")

# Create tabs
tab1, tab2 = st.tabs(["🔍 Predict Fish Type", "📖 Fish Details"])

with tab1:
    st.write("Upload a fish image and get its predicted category and confidence score.")

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

        # Show all class probabilities
        st.markdown("### 📊 Confidence Scores")
        for i, prob in enumerate(predictions):
            st.write(f"{class_names[i]}: {prob*100:.2f}%")

with tab2:
    st.write("Enter the fish type to get details from Google.")

    fish_name = st.text_input("Fish Name", placeholder="e.g., salmon, sea bass, trout")
    if st.button("Get Details"):
        if fish_name.strip():
            try:
                search_url = f"https://www.google.com/search?q={fish_name}+fish"
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(search_url, headers=headers)
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract snippet (Google search info box alternative)
                description = ""
                for span in soup.find_all("span"):
                    if len(span.text) > 50 and fish_name.lower() in span.text.lower():
                        description = span.text
                        break

                if description:
                    st.markdown(f"### ℹ️ Details about {fish_name}")
                    st.write(description)
                else:
                    st.warning("No detailed description found. Try another name.")
            except Exception as e:
                st.error(f"Error fetching details: {e}")
        else:
            st.warning("Please enter a fish name.")
