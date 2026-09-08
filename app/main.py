import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown

# Configure page
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")

working_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(working_dir, "trained_model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "plant_disease_prediction_model.keras")

# Auto-download model weights from Google Drive if not found locally
if not os.path.exists(model_path):
    # Extracted from your Google Drive link
    file_id = "1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"
    drive_url = f"https://drive.google.com/uc?id={file_id}"
    with st.spinner("Downloading model weights... This may take a minute on initial setup."):
        gdown.download(drive_url, model_path, quiet=False)


# Cache model in memory to prevent reloading per interaction
@st.cache_resource
def load_trained_model(path):
    return tf.keras.models.load_model(path)


model = load_trained_model(model_path)

# Load class label indices
labels_path = os.path.join(working_dir, "class_indices.json")
with open(labels_path, "r") as f:
    raw_indices = json.load(f)

# Handle both key-value configurations: {"0": "Class_Name"} or {"Class_Name": 0}
sample_val = next(iter(raw_indices.values()))
if isinstance(sample_val, int):
    class_indices = {str(v): k for k, v in raw_indices.items()}
else:
    class_indices = {str(k): v for k, v in raw_indices.items()}


def load_and_preprocess_image(image_file, target_size=(224, 224)):
    # Convert image strictly to 3-channel RGB (handles RGBA or Grayscale)
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)
    
    # Scale to [0, 1] matching model training normalization
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension: shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image_class(model, image_file, class_indices):
    preprocessed_img = load_and_preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)[0]
    
    # Extract top 3 prediction candidates
    top_3_indices = np.argsort(predictions)[-3:][::-1]
    results = []
    for idx in top_3_indices:
        label = class_indices.get(str(idx), "Unknown Class")
        confidence = float(predictions[idx] * 100)
        results.append((label, confidence))
    return results


# Application UI
st.title("🌿 Plant Disease Classifier")
st.write("Upload an image of a plant leaf to identify its health status and detect diseases.")

uploaded_image = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_image, caption="Uploaded Leaf", use_container_width=True)
    
    with col2:
        if st.button("Classify Leaf"):
            with st.spinner("Analyzing image..."):
                predictions = predict_image_class(model, uploaded_image, class_indices)
                top_class, top_conf = predictions[0]

                st.success(f"**Top Prediction:** {top_class}")
                st.info(f"**Confidence:** {top_conf:.2f}%")

                with st.expander("View Top Alternative Predictions"):
                    for label, conf in predictions[1:]:
                        st.write(f"- **{label}**: {conf:.2f}%")