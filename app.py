"""
app.py
------
Streamlit web app for the Face Recognition system.
Upload an image, the model predicts who it is (or "Unknown Person"
if confidence is below the threshold).
"""

import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model/face_recognition_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"
IMG_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.75

st.set_page_config(
    page_title="Face Recognition AI",
    page_icon="🧠",
    layout="centered",
)


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names


model, CLASS_NAMES = load_model_and_classes()


def predict(image: Image.Image, threshold: float):
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    preds = model.predict(img_array, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    confidence = float(preds[best_idx])

    if confidence < threshold:
        label = "Unknown Person"
    else:
        label = CLASS_NAMES[best_idx]

    all_probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    return label, confidence, all_probs


st.title("🧠 Face Recognition AI")
st.write(
    "Upload a photo and the model will predict which registered person it is, "
    "or flag it as **Unknown Person** if it isn't confident enough."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLD, step=0.05,
        help="Predictions below this confidence are shown as 'Unknown Person'."
    )
    st.markdown("---")
    st.caption(f"Registered people: {', '.join(CLASS_NAMES)}")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Analyzing..."):
        label, confidence, all_probs = predict(image, threshold)

    with col2:
        if label == "Unknown Person":
            st.warning(f"### 🚫 {label}")
            st.write(f"Best guess was below the confidence threshold.")
        else:
            st.success(f"### ✅ {label}")
            st.write(f"Confidence: **{confidence:.1%}**")

        st.markdown("---")
        st.write("**All class probabilities:**")
        for name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            st.write(f"{name}: {prob:.1%}")
            st.progress(prob)
else:
    st.info("👆 Upload an image to get a prediction.")