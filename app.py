
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('model_penyakit_jantung.pkl')

# Title
st.title("Prediksi Penyakit Jantung")
st.write("Masukkan data pasien untuk memprediksi apakah memiliki penyakit jantung.")

# User input
age = st.number_input("Umur", min_value=1, max_value=120, value=50)
sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
cp = st.selectbox("Tipe Nyeri Dada (CP)", [0, 1, 2, 3])
trestbps = st.number_input("Tekanan Darah (Resting)", min_value=80, max_value=200, value=120)
chol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=240)
fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl", [0, 1])
restecg = st.selectbox("Hasil EKG Saat Istirahat", [0, 1, 2])
thalach = st.number_input("Detak Jantung Maksimum", min_value=60, max_value=250, value=150)
exang = st.selectbox("Nyeri Dada Saat Olahraga (Angina)", [0, 1])

# Convert input
sex_val = 1 if sex == "Laki-laki" else 0
input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg, thalach, exang]])

# Prediction
if st.button("Prediksi"):
    prediction = model.predict(input_data)
    result = "Punya penyakit jantung" if prediction[0] == 1 else "Tidak punya penyakit jantung"
    st.success(f"Hasil Prediksi: {result}")
