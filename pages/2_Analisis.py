import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from kaggle.api.kaggle_api_extended import KaggleApi
import tempfile
import os
import json

st.title("Analisis Data")

# Fungsi untuk mengunduh dataset dari Kaggle
def download_kaggle_dataset(dataset_name):
    try:
        # Cek apakah aplikasi berjalan di Streamlit Cloud
        if hasattr(st.secrets, "kaggle"):
            # Streamlit Cloud: Gunakan Secrets
            kaggle_json = {
                "username": st.secrets["kaggle"]["username"],
                "key": st.secrets["kaggle"]["key"]
            }
        else:
            # Lokal: Gunakan file kaggle.json dari lokasi manual
            kaggle_json_path = r"/home/appuser/.kaggle/kaggle.json"  # Pastikan lokasi kaggle.json benar
            with open(kaggle_json_path, "r") as f:
                kaggle_json = json.load(f)

        # Simpan file kaggle.json ke lokasi default yang dikenali oleh API Kaggle
        kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        kaggle_file = os.path.join(kaggle_dir, "kaggle.json")

        with open(kaggle_file, "w") as f:
            json.dump(kaggle_json, f)

        # Set permission file untuk keamanan (khusus Linux/MacOS)
        os.chmod(kaggle_file, 0o600)

        # Membuat objek API Kaggle dan autentikasi
        api = KaggleApi()
        api.authenticate()

        # Unduh dataset ke direktori sementara
        with tempfile.TemporaryDirectory() as tmp_dir:
            api.dataset_download_files(dataset_name, path=tmp_dir, unzip=True)

            # Cari file CSV dalam direktori sementara
            for file in os.listdir(tmp_dir):
                if file.endswith(".csv"):
                    data = pd.read_csv(os.path.join(tmp_dir, file))
                    st.session_state.data = data
                    st.success("Dataset berhasil diunduh dan dimuat.")
                    return data
            else:
                st.error("Tidak ada file CSV ditemukan dalam dataset Kaggle.")
                return None
    except Exception as e:
        st.error(f"Error autentikasi atau unduhan Kaggle: {e}")
        return None

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
            data = download_kaggle_dataset(dataset_name)
            if data is not None:
                st.session_state.data = data

# Menampilkan data jika sudah diunggah atau diunduh
if 'data' in st.session_state:
    st.write("Data Awal:", st.session_state.data.head())

    # Preprocessing Data
    st.subheader("Preprocessing Data")
    if 'Heart Disease' in st.session_state.data.columns:
        # Periksa dan encode label 'Heart Disease' jika perlu
        label_encoder = LabelEncoder()
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

    # Pembersihan NaN
    if st.session_state.data.isnull().sum().any():
        st.warning("Data mengandung nilai NaN. Melakukan pembersihan data.")
        st.session_state.data = st.session_state.data.dropna()
        st.write("Data setelah pembersihan NaN:", st.session_state.data.head())

    st.subheader("Data setelah Preprocessing")
    st.dataframe(st.session_state.data)

    # Pisahkan fitur (X) dan label (y)
    X = st.session_state.data.drop(columns=['Heart Disease'])  # Pastikan Heart Disease tidak ada
    y = st.session_state.data['Heart Disease']  # Label target

    # Cek jika ada NaN di y (label), dan tangani
    if y.isnull().any():
        st.warning("Data label mengandung NaN, melakukan pembersihan data label.")
        data = st.session_state.data.dropna(subset=['Heart Disease'])  # Hapus NaN pada label
        y = data['Heart Disease']  # Update y setelah pembersihan

    # Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi dan latih model Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Simpan model di session state
    st.session_state.model = model

    # Formulir untuk input data baru
    st.subheader("Formulir Input Data Baru")
    # Input dari pengguna
    age = st.slider("Umur:", 0, 100, 50)
    sex = st.radio("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    chest_pain = st.selectbox("Tipe Nyeri Dada", [1, 2, 3, 4])
    bp = st.slider("Tekanan Darah:", 50, 200, 120)
    cholesterol = st.slider("Kolesterol:", 100, 400, 200)
    fbs_over_120 = st.radio("Gula Darah Puasa > 120", ["Tidak", "Ya"])
    ekg_results = st.selectbox("Hasil EKG", [0, 1, 2])
    max_hr = st.slider("Detak Jantung Maksimal:", 60, 200, 150)
    exercise_angina = st.radio("Angina Saat Olahraga", ["Tidak", "Ya"])
    st_depression = st.slider("Depresi ST:", 0.0, 5.0, 1.0)
    slope_of_st = st.selectbox("Kemiringan ST", [1, 2, 3])
    num_vessels = st.selectbox("Jumlah Pembuluh Darah Fluro", [0, 1, 2, 3])
    thallium = st.selectbox("Hasil Tes Thallium", [3, 6, 7])

    # Validasi input sebelum ditambahkan
    input_data = {
        "age": age,
        "sex": sex,
        "chest_pain": chest_pain,
        "bp": bp,
        "cholesterol": cholesterol,
        "fbs_over_120": fbs_over_120,
        "ekg_results": ekg_results,
        "max_hr": max_hr,
        "exercise_angina": exercise_angina,
        "st_depression": st_depression,
        "slope_of_st": slope_of_st,
        "num_vessels": num_vessels,
        "thallium": thallium
    }

    if all(v is not None for v in input_data.values()):
        # Mapping input ke nilai numerik
        sex = 1 if sex == "Laki-laki" else 0
        fbs_over_120 = 1 if fbs_over_120 == "Ya" else 0
        exercise_angina = 1 if exercise_angina == "Ya" else 0

        # DataFrame dari input pengguna tanpa kolom 'Heart Disease'
        user_data = pd.DataFrame([[age, sex, chest_pain, bp, cholesterol, fbs_over_120, 
                                   ekg_results, max_hr, exercise_angina, st_depression, 
                                   slope_of_st, num_vessels, thallium]], 
                                 columns=X.columns)

        # Prediksi untuk data baru
        prediction = model.predict(user_data)
        predicted_label = prediction[0]

        # Menambahkan hasil prediksi sebagai kolom 'Heart Disease'
        user_data['Heart Disease'] = predicted_label

        st.write("Prediksi Penyakit Jantung:", "Presence" if predicted_label == 1 else "Absence")
