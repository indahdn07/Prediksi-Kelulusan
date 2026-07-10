import os
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset_kelulusan_mahasiswa.csv")

LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
BANNER_PATH = os.path.join(ASSETS_DIR, "banner.png")

FITUR_LIST = [
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
TARGET_COL = "Status Kelulusan"


# ----------------------------------------------------------------------
# HELPER
# ----------------------------------------------------------------------
@st.cache_data
def load_dataset(path: str):
    """Load dataset kalau file tersedia. Return None kalau tidak ada."""
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Dataset ditemukan tapi gagal dibaca: {e}")
    return None


# ----------------------------------------------------------------------
# HEADER / BANNER
# ----------------------------------------------------------------------
if os.path.exists(BANNER_PATH):
    st.image(BANNER_PATH)

col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH)
    else:
        st.markdown("### 🎓")
with col_title:
    st.title("Prediksi Kelulusan Mahasiswa")
    st.caption("Aplikasi klasifikasi status kelulusan mahasiswa berbasis Machine Learning")

st.divider()

# ----------------------------------------------------------------------
# DESKRIPSI SINGKAT
# ----------------------------------------------------------------------
st.markdown(
    """
    Selamat datang di aplikasi **Prediksi Kelulusan Mahasiswa** 👋

    Aplikasi ini menggunakan model *Machine Learning* (Logistic Regression, Random Forest, dan XGBoost)
    untuk memprediksi apakah status kelulusan seorang mahasiswa adalah **Ya** atau **Tidak**
    berdasarkan data akademik seperti IPK, IPS, jumlah cuti akademik, kehadiran, dan lainnya.

    """
)

nav_col1, nav_col2 = st.columns(2)
with nav_col1:
    st.markdown(
        """
        - 📄 **Dataset** — melihat dan mengeksplorasi data mentah
        - 📊 **Visualisasi** — korelasi antar fitur & distribusi target
        - 🔮 **Prediksi** — input data mahasiswa baru & lihat hasil prediksi
        """
    )
with nav_col2:
    st.markdown(
        """
        - 📈 **Evaluasi** — akurasi, precision, recall, confusion matrix
        - ℹ️ **Tentang** — informasi project & tim
        """
    )

st.divider()

# ----------------------------------------------------------------------
# RINGKASAN DATASET (kalau tersedia)
# ----------------------------------------------------------------------
st.subheader("📌 Ringkasan Dataset")

df = load_dataset(DATASET_PATH)

if df is not None:
    total_data = len(df)

    if TARGET_COL in df.columns:
        status_ya = int((df[TARGET_COL] == 1).sum())
        status_tidak = int((df[TARGET_COL] == 0).sum())
    else:
        status_ya, status_tidak = None, None

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Data Mahasiswa", f"{total_data:,}")
    if status_ya is not None:
        m2.metric("Kelulusan: Ya", f"{status_ya:,}")
        m3.metric("Kelulusan: Tidak", f"{status_tidak:,}")

    with st.expander("Lihat contoh data (5 baris pertama)"):
        # Ambil 5 baris pertama
        df_tampil = df.head().copy()
        
        # Ubah angka 1 jadi "Ya" dan 0 jadi "Tidak" khusus untuk tampilan di tabel
        if TARGET_COL in df_tampil.columns:
            df_tampil[TARGET_COL] = df_tampil[TARGET_COL].map({1: "Ya", 0: "Tidak"})
            
        st.dataframe(df_tampil, width="stretch")
else:
    st.info(
        "Dataset belum ditemukan di `dataset/dataset_kelulusan_mahasiswa.csv`. "
        "Letakkan file dataset di folder tersebut agar ringkasan ini tampil."
    )

# ----------------------------------------------------------------------
# FITUR YANG DIGUNAKAN MODEL
# ----------------------------------------------------------------------
with st.expander("🧩 Fitur yang digunakan model"):
    st.write(", ".join(FITUR_LIST))
    st.caption(f"Target/label: **{TARGET_COL}** (0 = Tidak, 1 = Ya)")

# ----------------------------------------------------------------------
# STATUS MODEL
# ----------------------------------------------------------------------
st.subheader("🤖 Status Model")

# Sudah diganti jadi xgboost.json
models_info = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.json", 
}
model_cols = st.columns(len(models_info))
models_dir = os.path.join(BASE_DIR, "models")

for col, (name, filename) in zip(model_cols, models_info.items()):
    path = os.path.join(models_dir, filename)
    with col:
        if os.path.exists(path):
            st.success(f"✅ {name}\n\nSiap digunakan")
        else:
            st.error(f"❌ {name}\n\nFile `{filename}` belum ditemukan")

st.divider()
st.caption("Dibuat dengan Streamlit • Project Prediksi Kelulusan Mahasiswa")