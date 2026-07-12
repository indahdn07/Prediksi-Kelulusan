"""
pages/2_Visualisasi.py
Halaman visualisasi: distribusi target, korelasi antar fitur, dan
distribusi tiap fitur numerik terhadap status kelulusan.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import FEATURES, TARGET, encode_dataframe  # noqa: E402

st.set_page_config(page_title="Visualisasi", page_icon="📊", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset_kelulusan_mahasiswa.csv")

st.title("📊 Visualisasi Data")


@st.cache_data
def load_dataset(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


df = load_dataset(DATASET_PATH)

if df is None:
    st.error(f"Dataset tidak ditemukan di `{DATASET_PATH}`.")
    st.stop()

df_encoded = encode_dataframe(df)

# ----------------------------------------------------------------------
# DISTRIBUSI TARGET
# ----------------------------------------------------------------------
st.subheader("🎯 Distribusi Status Kelulusan")

if TARGET in df.columns:
    counts = df[TARGET].value_counts().sort_index()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(counts.rename("Jumlah"), use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="Blues_d")
        ax.set_xlabel("Status Kelulusan (0=Tidak Lulus, 1=Lulus)")
        ax.set_ylabel("Jumlah")
        ax.set_title("Distribusi Status Kelulusan")
        st.pyplot(fig)
else:
    st.warning(f"Kolom target '{TARGET}' tidak ditemukan di dataset.")

st.divider()

# ----------------------------------------------------------------------
# HEATMAP KORELASI
# ----------------------------------------------------------------------
st.subheader("🔥 Correlation Matrix")

numeric_df = df_encoded.select_dtypes(include="number")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="Blues",
    fmt=".2f",
    ax=ax,
)
ax.set_title("Correlation Matrix")
st.pyplot(fig)

st.divider()

# ----------------------------------------------------------------------
# DISTRIBUSI FITUR NUMERIK vs TARGET
# ----------------------------------------------------------------------
st.subheader("📈 Distribusi Fitur terhadap Status Kelulusan")

numeric_features = [f for f in FEATURES if f in numeric_df.columns and df_encoded[f].nunique() > 2]

selected_feature = st.selectbox("Pilih fitur", numeric_features)

if selected_feature and TARGET in df_encoded.columns:
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=df_encoded,
        x=selected_feature,
        hue=TARGET,
        kde=True,
        palette=["#f28e8e", "#4C72B0"],
        ax=ax2,
    )
    ax2.set_title(f"Distribusi {selected_feature} berdasarkan Status Kelulusan")
    st.pyplot(fig2)
