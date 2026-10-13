import os
import json
import urllib.request
from datetime import datetime

from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9ca3af;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .severity-high {
        background-color: rgba(255, 75, 75, 0.12);
        border-left: 5px solid #ff4b4b;
    }
    .severity-moderate {
        background-color: rgba(255, 165, 0, 0.12);
        border-left: 5px solid #ffa500;
    }
    .severity-none {
        background-color: rgba(33, 197, 93, 0.12);
        border-left: 5px solid #21c55d;
    }
    .disease-name {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .confidence-badge {
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        background-color: rgba(255,255,255,0.08);
        display: inline-block;
    }
    .info-label {
        font-weight: 700;
        margin-top: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Paths & Direct Release Download URL
# ----------------------------------------------------------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(working_dir, "trained_model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "plant_disease_prediction_model.keras")
labels_path = os.path.join(working_dir, "class_indices.json")
disease_info_path = os.path.join(working_dir, "disease_info.json")

MODEL_RELEASE_URL = "https://github.com/Kunal9122/plant-disease-prediction/releases/download/v1.0/plant_disease_prediction_model.keras"
SEVERITY_ICON = {"high": "🔴", "moderate": "🟠", "none": "🟢", "unknown": "⚪"}


# ----------------------------------------------------------------------------
# Model Download and Caching
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model(path, url):
    if not os.path.exists(path):
        progress_text = st.empty()
        download_bar = st.progress(0)
        progress_text.info("📥 Downloading model weights from GitHub Release (~547 MB). Please wait...")

        def update_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(int(block_num * block_size * 100 / total_size), 100)
                download_bar.progress(percent)

        try:
            urllib.request.urlretrieve(url, path, reporthook=update_progress)
            progress_text.empty()
            download_bar.empty()
        except Exception as e:
            progress_text.error(f"Failed to download model weights: {e}")
            raise e

    return tf.keras.models.load_model(path, compile=False)


model = get_model(model_path, MODEL_RELEASE_URL)

# ----------------------------------------------------------------------------
# Load class indices
# ----------------------------------------------------------------------------
with open(labels_path, "r") as f:
    raw_indices = json.load(f)

sample_val = next(iter(raw_indices.values()))
if isinstance(sample_val, int):
    class_indices = {str(v): k for k, v in raw_indices.items()}
else:
    class_indices = {str(k): v for k, v in raw_indices.items()}


# ----------------------------------------------------------------------------
# Load disease & pesticide database
# ----------------------------------------------------------------------------
@st.cache_data
def load_disease_info(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


disease_info_db = load_disease_info(disease_info_path)


def get_treatment(predicted_class):
    return disease_info_db.get(
        predicted_class,
        {
            "disease_name": predicted_class.replace("___", " - ").replace("_", " "),
            "severity": "unknown",
            "pesticide": "No specific record found. Consult an agricultural extension specialist.",
            "organic_alternative": "Ensure optimal aeration, avoid overhead irrigation, and prune affected leaves.",
            "prevention": "Sanitize tools, practice crop rotation, and inspect leaves regularly.",
            "symptoms": "Leaf spots, chlorosis, lesions, or wilting.",
        },
    )


# ----------------------------------------------------------------------------
# Preprocessing and Prediction
# ----------------------------------------------------------------------------
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_image_class(trained_model, image_file, mapping):
    preprocessed_img = load_and_preprocess_image(image_file)
    predictions = trained_model.predict(preprocessed_img)[0]

    top_3_indices = np.argsort(predictions)[-3:][::-1]
    return [(mapping.get(str(idx), "Unknown Class"), float(predictions[idx] * 100)) for idx in top_3_indices]


# ----------------------------------------------------------------------------
# Session State Setup
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "predictions_cache" not in st.session_state:
    st.session_state.predictions_cache = {}

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 About")
    st.write(
        "Upload leaf photographs to identify health conditions, "
        "detect plant diseases, and obtain organic & chemical treatment options."
    )

    st.markdown("---")
    st.markdown("### 🎨 Severity Legend")
    st.markdown("🔴 High — critical damage risk")
    st.markdown("🟠 Moderate — treat promptly")
    st.markdown("🟢 None — foliage is healthy")

    st.markdown("---")
    st.markdown("### 📜 Prediction History")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.caption(f"{item['time']} — {item['disease']} ({item['confidence']:.1f}%)")
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.predictions_cache = {}
            st.rerun()
    else:
        st.caption("No classifications yet.")

# ----------------------------------------------------------------------------
# Main Page Header
# ----------------------------------------------------------------------------
st.markdown('<p class="main-title">🌿 Plant Disease Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload one or multiple leaf images for diagnostic evaluation and mitigation steps.</p>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# File Uploader
# ----------------------------------------------------------------------------
uploaded_images = st.file_uploader(
    "Choose leaf image(s)...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_images:
    for uploaded_image in uploaded_images:
        st.markdown("---")
        col1, col2 = st.columns([1, 1.4])

        with col1:
            st.image(uploaded_image, caption=uploaded_image.name, use_container_width=True)
            if st.button(f"Classify: {uploaded_image.name}", key=f"btn_{uploaded_image.name}", use_container_width=True):
                with st.spinner("Analyzing foliage patterns..."):
                    preds = predict_image_class(model, uploaded_image, class_indices)
                    top_cls, top_cnf = preds[0]
                    treat_info = get_treatment(top_cls)

                    st.session_state.predictions_cache[uploaded_image.name] = {
                        "predictions": preds,
                        "info": treat_info,
                        "top_conf": top_cnf,
                    }

                    st.session_state.history.append(
                        {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "disease": treat_info["disease_name"],
                            "confidence": top_cnf,
                        }
                    )

        with col2:
            if uploaded_image.name in st.session_state.predictions_cache:
                cached = st.session_state.predictions_cache[uploaded_image.name]
                info = cached["info"]
                top_conf = cached["top_conf"]
                predictions = cached["predictions"]

                severity = info.get("severity", "unknown")
                severity_class = f"severity-{severity}" if severity in ("high", "moderate", "none") else "severity-none"

                st.markdown(
                    f"""
                    <div class="result-card {severity_class}">
                        <div class="disease-name">{SEVERITY_ICON.get(severity, '⚪')} {info['disease_name']}</div>
                        <span class="confidence-badge">Confidence: {top_conf:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(top_conf / 100, 1.0))

                with st.expander("🔍 Symptoms", expanded=True):
                    st.write(info.get("symptoms", "N/A"))
                with st.expander("💊 Recommended Pesticide", expanded=True):
                    st.write(info.get("pesticide", "N/A"))
                with st.expander("🌱 Organic Alternative"):
                    st.write(info.get("organic_alternative", "N/A"))
                with st.expander("🛡️ Prevention Tips"):
                    st.write(info.get("prevention", "N/A"))

                st.markdown('<p class="info-label">Top 3 Predictions</p>', unsafe_allow_html=True)
                for label, conf in predictions:
                    clean_label = label.replace("___", " - ").replace("_", " ")
                    st.write(f"{clean_label}: {conf:.1f}%")
                    st.progress(min(conf / 100, 1.0))

                st.markdown('<p class="info-label">Was this prediction accurate?</p>', unsafe_allow_html=True)
                fb_col1, fb_col2 = st.columns(2)
                if fb_col1.button("👍 Yes", key=f"yes_{uploaded_image.name}"):
                    st.success("Feedback recorded. Thank you!")
                if fb_col2.button("👎 No", key=f"no_{uploaded_image.name}"):
                    st.info("Feedback noted for ongoing model calibration.")
else:
    st.info("👆 Upload one or more leaf images to get started.")