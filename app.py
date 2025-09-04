import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input
from huggingface_hub import hf_hub_download

st.title("🍔🥗🍣 Food-101 Image Classification")

model_path = hf_hub_download(
    repo_id="zahratalitha/101food",  
    filename="food101_best.h5"           
)
model = tf.keras.models.load_model(model_path, compile=False)
st.write("✅ Model berhasil dimuat")
st.write("Input shape model:", model.input_shape)

CLASS_NAMES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque",
    "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai",
    "paella", "pancakes", "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop",
    "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli",
    "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops", "seaweed_salad",
    "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls",
    "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
    "tuna_tartare", "waffles"
]

# Preprocessing
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np

def preprocess(img):
    # Pastikan 3 channel
    if img.mode != "RGB":
        img = img.convert("RGB")
    target_size = tuple(model.input_shape[1:3])
    img = img.resize(target_size)
    img_array = np.asarray(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


uploaded_file = st.file_uploader("Upload gambar makanan:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    input_img = preprocess(image)
    pred = model.predict(input_img)

    class_idx = int(np.argmax(pred[0]))
    label = CLASS_NAMES[class_idx]
    confidence = float(np.max(pred[0]))

    st.subheader(f"🍽 Prediksi: {label.replace('_', ' ').title()}")
    st.write(f"Confidence: {confidence:.2f}")
