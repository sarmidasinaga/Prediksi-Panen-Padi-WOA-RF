import streamlit as st
import pandas as pd
import pickle

# Load model
with open('final_woa_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Prediksi Hasil Panen')

# Form input sesuai fitur model
# Contoh: jika model butuh fitur 'luas_lahan', 'curah_hujan', dst.
luas_lahan = st.number_input('Luas Lahan (ha)', min_value=0.0)
curah_hujan = st.number_input('Curah Hujan (mm)', min_value=0.0)
# Tambahkan input lain sesuai fitur model

if st.button('Prediksi'):
    # Ubah ke DataFrame (urutan kolom harus sama dengan model training)
    data = pd.DataFrame([[luas_lahan, curah_hujan]], columns=['luas_lahan', 'curah_hujan'])
    hasil_prediksi = model.predict(data)
    st.success(f'Prediksi hasil panen: {hasil_prediksi[0]:,.2f} ton')
