import streamlit as st
import pandas as pd
import os
import kaggle
import tempfile
import json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Mendapatkan kredensial dari Streamlit secrets
username = st.secrets["kaggle_json"]["username"]
key = st.secrets["kaggle_json"]["key"]

# Set variabel lingkungan
os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = key

# Verifikasi bahwa variabel lingkungan sudah terisi dengan benar
st.write("Kaggle Username:", os.getenv("KAGGLE_USERNAME"))
st.write("Kaggle Key:", os.getenv("KAGGLE_KEY"))

# Autentikasi dengan API Kaggle
kaggle.api.authenticate()

st.title("Input Data")

# Pilihan metode input
input_method = st.radio("Pilih metode input data:", ("Unggah file CSV", "Nama dataset Kaggle"))

# Jika metode input adalah file upload
if input_method == "Unggah file CSV":
    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

    if uploaded_file is not None:
        # Load dataset
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data

# Jika metode input adalah Kaggle
elif input_method == "Nama dataset Kaggle":
    kaggle_username = st.text_input("Masukkan nama/username Kaggle dataset:")
    kaggle_dataset = st.text_input("Masukkan nama dataset Kaggle (contoh: 'dataset-name'):") 

    if st.button("Unduh dataset"):
        if kaggle_username and kaggle_dataset:
            try:
                # Mengambil API key dari secrets Streamlit
                kaggle_json = st.secrets["kaggle_json"]
                username = kaggle_json["username"]
                key = kaggle_json["key"]

                # Debugging: Menampilkan API key dan username yang digunakan
                st.write(f"API Username: {username}")
                st.write(f"API Key: {key}")

                # Set lingkungan API Kaggle
                os.environ["KAGGLE_USERNAME"] = username
                os.environ["KAGGLE_KEY"] = key

                # Verifikasi autentikasi dengan API Kaggle
                kaggle.api.authenticate()  # Mengautentikasi dengan Kaggle API
                st.write("API Kaggle berhasil diautentikasi!")

                # Menggunakan direktori sementara untuk menyimpan file CSV
                with tempfile.TemporaryDirectory() as tmp_dir:
                    kaggle.api.dataset_download_files(f"{kaggle_username}/{kaggle_dataset}", path=tmp_dir, unzip=True)

                    # Debugging: Menampilkan isi direktori tempat file diunduh
                    st.write(f"Isi direktori sementara: {os.listdir(tmp_dir)}")

                    # Cari file CSV dalam direktori sementara dan memuat ke dalam DataFrame
                    for file in os.listdir(tmp_dir):
                        if file.endswith(".csv"):
                            st.write(f"Menemukan file CSV: {file}")  # Debug: Menampilkan nama file CSV
                            data = pd.read_csv(f"{tmp_dir}/{file}")
                            st.session_state.data = data
                            break
            except Exception as e:
                st.error(f"Gagal mengunduh dataset: {str(e)}")

# Menampilkan data jika sudah diunggah atau diunduh
if 'data' in st.session_state:
    st.write("Data Awal:", st.session_state.data.head())

    # Preprocessing Data
    st.subheader("Preprocessing Data")
    label_encoder = LabelEncoder()
    if 'Heart Disease' in st.session_state.data.columns:
        st.session_state.data['Heart Disease'] = label_encoder.fit_transform(st.session_state.data['Heart Disease'])

    # Pilihan untuk Normalisasi
    st.write("Pilih kolom untuk normalisasi:")
    numeric_cols = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns
    cols_to_normalize = st.multiselect("Kolom Fitur Numerik:", numeric_cols)

    if cols_to_normalize:
        st.write(f"Melakukan normalisasi pada kolom: {cols_to_normalize}")
        scaler = MinMaxScaler()
        st.session_state.data[cols_to_normalize] = scaler.fit_transform(st.session_state.data[cols_to_normalize])
        st.write("Data setelah normalisasi:", st.session_state.data.head())

    # Pembersihan data jika ada NaN
    if st.session_state.data.isnull().sum().any():
        st.warning("Data mengandung nilai NaN. Melakukan pembersihan data.")
        st.session_state.data = st.session_state.data.dropna()
        st.write("Data setelah pembersihan NaN:", st.session_state.data.head())

    st.subheader("Data setelah Preprocessing")
    st.dataframe(st.session_state.data)
