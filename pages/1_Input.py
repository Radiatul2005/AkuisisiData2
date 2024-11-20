import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from kaggle.api.kaggle_api_extended import KaggleApi
import tempfile
import os

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
    dataset_name = st.text_input("Masukkan nama dataset Kaggle (contoh: 'username/dataset-name'):")

    if st.button("Unduh dataset"):
        if dataset_name:
            # Autentikasi dengan Secrets
            kaggle_username = st.secrets["kaggle"]["username"]
            kaggle_key = st.secrets["kaggle"]["key"]

            # Membuat objek API Kaggle
            api = KaggleApi()
            api.authenticate()

            # Gunakan direktori sementara untuk menyimpan file CSV sementara
            with tempfile.TemporaryDirectory() as tmp_dir:
                api.dataset_download_files(dataset_name, path=tmp_dir, unzip=True)

                # Cari file CSV dalam direktori sementara dan memuat ke dalam DataFrame
                for file in os.listdir(tmp_dir):
                    if file.endswith(".csv"):
                        data = pd.read_csv(os.path.join(tmp_dir, file))
                        st.session_state.data = data
                        break

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

    if st.session_state.data.isnull().sum().any():
        st.warning("Data mengandung nilai NaN. Melakukan pembersihan data.")
        st.session_state.data = st.session_state.data.dropna()
        st.write("Data setelah pembersihan NaN:", st.session_state.data.head())

    st.subheader("Data setelah Preprocessing")
    st.dataframe(st.session_state.data)
