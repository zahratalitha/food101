import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121

# -----------------------------
# Judul
# -----------------------------
st.title("🍔🥗🍣 Food-101 Image Classification (DenseNet121)")

# -----------------------------
# Download & Load Model
# -----------------------------
st.write("📥 Loading model dari HuggingFace...")
model_path = hf_hub_download(
    repo_id="zahratalitha/101food",
    filename="food101_best.h5"
)

model = tf.keras.models.load_model(
    model_path,
    compile=False,
    custom_objects={"DenseNet121": DenseNet121}
)
st.success("✅ Model berhasil dimuat!")

st.write("Input shape model:", model.input_shape)
st.write("Output shape model:", model.output_shape)

# -----------------------------
# Sidebar: Upload Gambar
# -----------------------------
st.sidebar.header("📤 Upload Gambar")
uploaded_file = st.sidebar.file_uploader("Pilih gambar makanan", type=["jpg", "jpeg", "png"])

# -----------------------------
# Preprocessing DenseNet
# -----------------------------
IMG_HEIGHT, IMG_WIDTH = 224, 224  # Sesuai DenseNet121

def preprocess_image_densenet(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # preprocessing DenseNet
    return img_array

# -----------------------------
# Prediksi
# -----------------------------
if uploaded_file is not None:
    # Tampilkan gambar di halaman utama
    st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)

    # Preprocess & prediksi
    img_array = preprocess_image_densenet(uploaded_file)
    preds = model.predict(img_array)

    # Ambil top-5 prediksi
    top5_idx = preds[0].argsort()[-5:][::-1]
    top5_probs = preds[0][top5_idx]

    # Jika ada file labels.txt, gunakan untuk nama kelas
    try:
        with open("labels.txt", "r") as f:
            class_labels = [line.strip() for line in f.readlines()]
        top5_labels = [class_labels[i] for i in top5_idx]
    except:
        top5_labels = [f"Kelas {i}" for i in top5_idx]

    # Tampilkan hasil
    st.subheader("📌 Top 5 Prediksi")
    for label, prob in zip(top5_labels, top5_probs):
        st.write(f"{label}: {prob*100:.2f}%")
