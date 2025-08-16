import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load model dan scaler
model = joblib.load("model_knn_nanas.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Prediksi Kematangan Nanas", page_icon="🍍", layout="wide")

st.markdown("<h1 style='text-align: center; color: green;'>🍍 Prediksi Kematangan Buah Nanas 🍍</h1>", unsafe_allow_html=True)
st.write("Unggah gambar nanas atau gambar lain, sistem akan memprediksi tingkat kematangan atau memberi tahu jika gambar tidak dikenali.")

# Fungsi ekstraksi RGB rata-rata
def get_average_rgb(img):
    img_array = np.array(img)
    if img_array.ndim == 3:  # RGB
        R = np.mean(img_array[:, :, 0])
        G = np.mean(img_array[:, :, 1])
        B = np.mean(img_array[:, :, 2])
        return R, G, B
    else:
        return 0, 0, 0

# Fungsi prediksi
def prediksi_rgb(r, g, b):
    rgb_scaled = scaler.transform([[r, g, b]])
    pred = model.predict(rgb_scaled)[0]
    return pred

# Layout 2 kolom utama
col1, col2 = st.columns([1.2, 1])

with col1:
    uploaded_file = st.file_uploader("📤 Unggah gambar buah nanas", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # Ekstraksi RGB
        R, G, B = get_average_rgb(image)

        # Prediksi
        hasil_prediksi = prediksi_rgb(R, G, B)

        # Card Prediksi
        if hasil_prediksi == "tidak_dikenal":
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#8B0000; color:white; margin-bottom:10px;">
                    <h4>🚫 Gambar tidak dikenali</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#006400; color:white; margin-bottom:10px;">
                    <h4>✅ Prediksi: {hasil_prediksi.upper()}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Card RGB
        st.markdown(
            f"""
            <div style="padding:10px; border-radius:10px; background-color:#0d1b2a; color:white;">
                <b>🎨 Nilai RGB:</b><br>
                🔴 R = {R:.2f}<br>
                🟢 G = {G:.2f}<br>
                🔵 B = {B:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

with col2:
    if 'image' in locals():
        st.image(image, caption="📷 Gambar yang diunggah", width=300)
    else:
        st.info("Belum ada gambar diunggah")
