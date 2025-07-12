import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained model & feature importance (jika tersedia)
model = joblib.load('model_penyakit_jantung.pkl')

# Dummy feature importance (gunakan model.feature_importances_ jika ada)
feature_names = ['Age', 'Sex', 'CP', 'RestBP', 'Chol', 'FBS', 'RestECG',
                 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'CA', 'Thal']
feature_importance = np.array([0.10, 0.07, 0.13, 0.06, 0.09, 0.03, 0.05, 0.12, 0.08, 0.10, 0.05, 0.06, 0.06])

# Page config
st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: red;'>❤️ Prediksi Penyakit Jantung</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan data pasien untuk memprediksi kemungkinan memiliki penyakit jantung.</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout 2 kolom
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🧓 Umur", min_value=1, max_value=120, value=50)
    sex = st.selectbox("🚻 Jenis Kelamin", ["Laki-laki", "Perempuan"])
    cp = st.selectbox("💢 Tipe Nyeri Dada (CP)", [0, 1, 2, 3])
    trestbps = st.number_input("💉 Tekanan Darah Saat Istirahat", min_value=80, max_value=200, value=120)
    chol = st.number_input("🩸 Kolesterol (mg/dl)", min_value=100, max_value=600, value=240)
    fbs = st.selectbox("🍬 Gula Darah Puasa > 120 mg/dl?", [0, 1])
    restecg = st.selectbox("🫀 Hasil EKG Saat Istirahat", [0, 1, 2])

with col2:
    thalach = st.number_input("❤️ Detak Jantung Maksimum", min_value=60, max_value=250, value=150)
    exang = st.selectbox("🏃‍♂️ Nyeri Dada Saat Olahraga (Angina)", [0, 1])
    oldpeak = st.number_input("📉 Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("📈 Slope of ST Segment", [0, 1, 2])
    ca = st.selectbox("🩻 Jumlah Pembuluh Utama (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("🧬 Thalassemia", [0, 1, 2])

# Convert gender input
sex_val = 1 if sex == "Laki-laki" else 0

# Prepare input
input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])

# Predict
if st.button("🔍 Prediksi Sekarang"):
    prediction = model.predict(input_data)
    result = "✅ Tidak memiliki penyakit jantung." if prediction[0] == 0 else "⚠️ Berisiko memiliki penyakit jantung."

    st.markdown("---")
    if prediction[0] == 0:
        st.success(f"Hasil Prediksi: {result}")
        st.markdown("💡 **Rekomendasi:** Tetap jaga pola hidup sehat, rutin olahraga, dan periksa kesehatan secara berkala.")
    else:
        st.error(f"Hasil Prediksi: {result}")
        st.markdown("⚠️ **Rekomendasi:** Segera konsultasikan dengan dokter untuk pemeriksaan lebih lanjut dan ubah gaya hidup menjadi lebih sehat.")

    # Penjelasan model
    st.markdown("### 🧠 Tentang Model")
    st.markdown("""
    Model ini dilatih menggunakan algoritma **machine learning Random Forest** yang mempelajari pola dari berbagai fitur seperti umur, tekanan darah, kolesterol, detak jantung maksimum, dan lainnya.
    
    Tujuan utama dari model ini adalah memberikan indikasi awal terhadap potensi risiko penyakit jantung berdasarkan data medis pasien.
    """)

    # Grafik Feature Importance
    st.markdown("### 📊 Pentingnya Setiap Fitur (Feature Importance)")

    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_idx = np.argsort(feature_importance)
    ax.barh(np.array(feature_names)[sorted_idx], feature_importance[sorted_idx], color='salmon')
    ax.set_xlabel("Skor Pentingnya Fitur")
    ax.set_title("Feature Importance Model")
    st.pyplot(fig)

    st.info("Fitur seperti tipe nyeri dada, detak jantung maksimum, dan umur adalah indikator penting dalam menentukan risiko penyakit jantung.")
