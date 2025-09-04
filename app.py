import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications import DenseNet121  # ✅ Tambahin ini

st.title("🍔🥗🍣 Food-101 Image Classification (DenseNet121)")

# === Download Model dari HuggingFace ===
model_path = hf_hub_download(
    repo_id="zahratalitha/101food",
    filename="food101_best.h5"
)

# === Load Model dengan custom_objects ===
st.write("📥 Loading model...")
model = tf.keras.models.load_model(
    model_path,
    compile=False,
    custom_objects={"DenseNet121": DenseNet121}  # ✅ Tambahin ini
)
st.success("✅ Model berhasil dimuat!")

st.write("Input shape model:", model.input_shape)
st.write("Output shape model:", model.output_shape)
