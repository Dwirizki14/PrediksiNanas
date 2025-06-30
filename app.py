import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

# Load model & scaler
model = joblib.load('model_knn.pkl')
scaler = joblib.load('scaler_knn.pkl')

# Load dataset untuk visualisasi
df = pd.read_csv('dataset_rgb.csv')

st.title("🍍 Klasifikasi Kematangan Buah Nanas")

st.write("Silakan upload gambar untuk diprediksi:")

# Sidebar: upload gambar
st.sidebar.header("Upload Gambar")
uploaded_file = st.sidebar.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

btn_predict = st.sidebar.button("Prediksi")

if btn_predict:
    if uploaded_file is not None:
        # Baca gambar & hitung RGB mean
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        r_mean = int(np.mean(img_array[:,:,0]))
        g_mean = int(np.mean(img_array[:,:,1]))
        b_mean = int(np.mean(img_array[:,:,2]))
        sample = [[r_mean, g_mean, b_mean]]

        st.image(image, caption=f"Gambar diupload (RGB mean: {r_mean}, {g_mean}, {b_mean})", width=200)

        # Scaling & prediksi
        sample_scaled = scaler.transform(sample)
        prediksi = model.predict(sample_scaled)[0]
        probs = model.predict_proba(sample_scaled)[0]
        max_prob = max(probs)

        # Hasil
        if prediksi == 'not_pineapple' or max_prob < 0.8:
            st.warning("Gambar tidak dikenal ❗")
        else:
            st.success(f"Hasil prediksi: **{prediksi}**")
    else:
        st.error("Mohon upload gambar terlebih dahulu.")
