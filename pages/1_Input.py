import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import kaggle
import tempfile
import os
import json

st.title("Input Data")

# Tentukan direktori .kaggle untuk Windows atau Linux
if os.name == 'nt':  # Windows
    kaggle_dir = os.path.join(os.getenv('USERPROFILE'), '.kaggle')
else:  # Linux atau Streamlit Cloud
    kaggle_dir = os.path.expanduser("~/.kaggle")

# Pastikan direktori .kaggle ada
os.makedirs(kaggle_dir, exist_ok=True)

# Mengambil file kaggle.json dari Streamlit Secrets
kaggle_json = st.secrets.get("kaggle")

if kaggle_json is None:
    st.error("File kaggle.json tidak ditemukan di Streamlit Secrets. Pastikan kamu sudah menambahkannya.")
else:
    # Menyimpan file kaggle.json ke direktori yang sesuai
    try:
        with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
            json.dump(kaggle_json, f)
        st.success(f"File kaggle.json berhasil disalin ke: {kaggle_dir}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menyalin kaggle.json: {e}")

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
                    # Autentikasi dengan API Kaggle
                    kaggle.api.authenticate()
                    st.success("Autentikasi API Kaggle berhasil!")
                    
                    # Menggunakan direktori sementara untuk menyimpan file CSV sementara
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        kaggle.api.dataset_download_files(f"{kaggle_username}/{kaggle_dataset}", path=tmp_dir, unzip=True)

                        # Cari file CSV dalam direktori sementara dan memuat ke dalam DataFrame
                        for file in os.listdir(tmp_dir):
                            if file.endswith(".csv"):
                                data = pd.read_csv(f"{tmp_dir}/{file}")
                                st.session_state.data = data
                                st.success("Dataset berhasil diunduh!")
                                break
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat mengunduh dataset Kaggle: {e}")

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

        # Menangani nilai NaN
        if st.session_state.data.isnull().sum().any():
            st.warning("Data mengandung nilai NaN. Pilih bagaimana cara menangani data kosong:")
            nan_handling_option = st.radio("Pilih cara menangani NaN:", ("Hapus baris yang mengandung NaN", "Isi NaN dengan nilai rata-rata"))
            
            if nan_handling_option == "Hapus baris yang mengandung NaN":
                st.session_state.data = st.session_state.data.dropna()
                st.write("Data setelah pembersihan NaN:", st.session_state.data.head())
            elif nan_handling_option == "Isi NaN dengan nilai rata-rata":
                st.session_state.data.fillna(st.session_state.data.mean(), inplace=True)
                st.write("Data setelah mengisi NaN dengan rata-rata:", st.session_state.data.head())

        st.subheader("Data setelah Preprocessing")
        st.dataframe(st.session_state.data)
