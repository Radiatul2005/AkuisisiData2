import streamlit as st
import os
import json
import tempfile
import pandas as pd
import kaggle

st.title("Input Data Kaggle")

# Tentukan direktori untuk file kaggle.json
if os.name == 'nt':  # Windows
    kaggle_dir = os.path.join(os.getenv('USERPROFILE'), '.kaggle')
else:  # Linux atau Streamlit Cloud
    kaggle_dir = os.path.expanduser("~/.kaggle")

os.makedirs(kaggle_dir, exist_ok=True)

# Ambil Secrets dan buat file kaggle.json
if "kaggle" in st.secrets:
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json_path, "w") as f:
        json.dump(st.secrets["kaggle"], f)

    # Set izin file
    os.chmod(kaggle_json_path, 0o600)
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_dir
    st.success("File kaggle.json berhasil disiapkan!")
else:
    st.error("Secrets Kaggle tidak ditemukan!")

# Input dataset Kaggle
kaggle_username = st.text_input("Masukkan nama pengguna Kaggle dataset:")
kaggle_dataset = st.text_input("Masukkan nama dataset Kaggle (contoh: 'dataset-name'):")

if st.button("Unduh dataset"):
    if kaggle_username and kaggle_dataset:
        try:
            kaggle.api.authenticate()
            with tempfile.TemporaryDirectory() as tmp_dir:
                kaggle.api.dataset_download_files(f"{kaggle_username}/{kaggle_dataset}", path=tmp_dir, unzip=True)
                st.success("Dataset berhasil diunduh!")

                # Cari file CSV
                for file in os.listdir(tmp_dir):
                    if file.endswith(".csv"):
                        data = pd.read_csv(os.path.join(tmp_dir, file))
                        st.dataframe(data.head())
                        break
        except Exception as e:
            st.error(f"Kesalahan saat mengunduh dataset: {e}")
    else:
        st.warning("Mohon isi username dan nama dataset Kaggle!")
