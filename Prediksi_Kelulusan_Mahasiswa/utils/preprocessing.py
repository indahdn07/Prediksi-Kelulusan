"""
utils/preprocessing.py

Fungsi-fungsi untuk menyiapkan data (encoding kategori, split fitur/target)
sebelum masuk ke model Machine Learning.

CATATAN / ASUMSI PENTING:
Karena file notebook Colab asli tidak bisa diakses langsung, mapping kategori
di bawah ini (PEKERJAAN_MAP, KEHADIRAN_MAP) adalah ASUMSI berdasarkan nama
kolom yang terlihat di screenshot. Sesuaikan urutan/isi mapping ini dengan
encoding yang benar-benar dipakai saat training model di notebook kamu,
supaya hasil prediksi konsisten dengan model yang sudah dilatih.
"""

import pandas as pd

# Urutan fitur HARUS sama persis dengan urutan X.columns saat training model
FEATURES = [
    "IPK",
    "Mata Kuliah Tidak Lulus",
    "Jumlah Cuti Akademik",
    "Pekerjaan Sambil Kuliah",
    "Jumlah Semester",
    "IPS Rata-rata",
    "IPS Semester Akhir",
    "IPS Tren",
    "Kategori Kehadiran",
]

TARGET = "Status Kelulusan"

# ASUMSI: kolom kategorikal & mapping ke angka (label encoding sederhana)
PEKERJAAN_MAP = {"Tidak": 0, "Ya": 1}
KEHADIRAN_MAP = {"Rendah": 0, "Sedang": 1, "Tinggi": 2}

PEKERJAAN_OPTIONS = list(PEKERJAAN_MAP.keys())
KEHADIRAN_OPTIONS = list(KEHADIRAN_MAP.keys())


def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode kolom kategorikal pada dataframe mentah menjadi numerik,
    mengikuti mapping di atas. Aman dipanggil walau kolom sudah numerik
    (misalnya dataset sudah di-encode sebelumnya).
    """
    df = df.copy()

    if "Pekerjaan Sambil Kuliah" in df.columns and df["Pekerjaan Sambil Kuliah"].dtype == object:
        df["Pekerjaan Sambil Kuliah"] = df["Pekerjaan Sambil Kuliah"].map(PEKERJAAN_MAP)

    if "Kategori Kehadiran" in df.columns and df["Kategori Kehadiran"].dtype == object:
        df["Kategori Kehadiran"] = df["Kategori Kehadiran"].map(KEHADIRAN_MAP)

    return df


def get_features_target(df: pd.DataFrame):
    """
    Encode dataframe, lalu pisahkan menjadi X (fitur) dan y (target).
    Hanya kolom di FEATURES yang diambil, dengan urutan yang konsisten.
    """
    df_encoded = encode_dataframe(df)

    missing = [c for c in FEATURES if c not in df_encoded.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di dataset: {missing}")

    X = df_encoded[FEATURES]
    y = df_encoded[TARGET] if TARGET in df_encoded.columns else None
    return X, y


def encode_single_input(input_dict: dict) -> pd.DataFrame:
    """
    Encode satu baris input dari form Streamlit (dict) menjadi DataFrame
    1 baris yang siap dipakai model.predict().

    input_dict contoh:
    {
        "IPK": 3.2,
        "Mata Kuliah Tidak Lulus": 1,
        "Jumlah Cuti Akademik": 0,
        "Pekerjaan Sambil Kuliah": "Tidak",
        "Jumlah Semester": 8,
        "IPS Rata-rata": 3.1,
        "IPS Semester Akhir": 3.4,
        "IPS Tren": 0.2,
        "Kategori Kehadiran": "Tinggi",
    }
    """
    row = dict(input_dict)
    row["Pekerjaan Sambil Kuliah"] = PEKERJAAN_MAP.get(
        row.get("Pekerjaan Sambil Kuliah"), row.get("Pekerjaan Sambil Kuliah")
    )
    row["Kategori Kehadiran"] = KEHADIRAN_MAP.get(
        row.get("Kategori Kehadiran"), row.get("Kategori Kehadiran")
    )

    df_row = pd.DataFrame([row])[FEATURES]
    return df_row
