"""
pages/1_Dataset.py
Halaman untuk menampilkan & mengeksplorasi dataset mentah.
"""

import os
import sys
import pandas as pd
import streamlit as st

# Supaya bisa import dari folder utils/ di root project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import FEATURES, TARGET  # noqa: E402

st.set_page_config(page_title="Dataset", page_icon="📄", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset_kelulusan_mahasiswa.csv")

st.title("📄 Dataset Kelulusan Mahasiswa")


@st.cache_data
def load_dataset(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


df = load_dataset(DATASET_PATH)

if df is None:
    st.error(
        "Dataset tidak ditemukan. Letakkan file di:\n\n"
        f"`{DATASET_PATH}`"
    )
    st.stop()

# ----------------------------------------------------------------------
# RINGKASAN
# ----------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah Baris", f"{df.shape[0]:,}")
c2.metric("Jumlah Kolom", df.shape[1])
c3.metric("Missing Values", int(df.isna().sum().sum()))
c4.metric("Data Duplikat", int(df.duplicated().sum()))

st.divider()

# ----------------------------------------------------------------------
# FILTER SEDERHANA
# ----------------------------------------------------------------------
st.subheader("🔍 Filter Data")

filter_cols = st.columns(3)
filtered_df = df.copy()

with filter_cols[0]:
    if TARGET in df.columns:
        status_options = ["Semua"] + sorted(df[TARGET].dropna().unique().tolist())
        status_filter = st.selectbox("Status Kelulusan", status_options)
        if status_filter != "Semua":
            filtered_df = filtered_df[filtered_df[TARGET] == status_filter]

with filter_cols[1]:
    if "IPK" in df.columns:
        ipk_min, ipk_max = float(df["IPK"].min()), float(df["IPK"].max())
        ipk_range = st.slider("Rentang IPK", ipk_min, ipk_max, (ipk_min, ipk_max))
        filtered_df = filtered_df[
            (filtered_df["IPK"] >= ipk_range[0]) & (filtered_df["IPK"] <= ipk_range[1])
        ]

with filter_cols[2]:
    n_rows = st.number_input("Jumlah baris ditampilkan", min_value=5, max_value=1000, value=50, step=5)

st.dataframe(filtered_df.head(int(n_rows)), use_container_width=True)

st.caption(f"Menampilkan {min(int(n_rows), len(filtered_df))} dari {len(filtered_df):,} baris terfilter.")

# ----------------------------------------------------------------------
# INFO TIPE DATA
# ----------------------------------------------------------------------
with st.expander("🧬 Info Tipe Data & Statistik Deskriptif"):
    st.write("**Tipe data tiap kolom:**")
    st.dataframe(df.dtypes.astype(str).rename("Tipe Data"), use_container_width=True)

    st.write("**Statistik deskriptif (kolom numerik):**")
    st.dataframe(df.describe(), use_container_width=True)

# ----------------------------------------------------------------------
# DOWNLOAD DATA HASIL FILTER
# ----------------------------------------------------------------------
csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download data terfilter (CSV)",
    data=csv_data,
    file_name="dataset_terfilter.csv",
    mime="text/csv",
)
