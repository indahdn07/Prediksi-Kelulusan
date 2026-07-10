"""
pages/4_Evaluasi.py
Halaman evaluasi performa model: accuracy, precision, recall, f1,
confusion matrix, classification report, dan feature importance.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import get_features_target, FEATURES, encode_dataframe  # noqa: E402
from utils.load_model import available_models, load_model  # noqa: E402

st.set_page_config(page_title="Evaluasi", page_icon="📈", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset_kelulusan_mahasiswa.csv")

# ASUMSI split — sesuaikan kalau berbeda dari notebook asli
TEST_SIZE = 0.2
RANDOM_STATE = 42

st.title("📈 Evaluasi Model")

with st.expander("ℹ️ Asumsi pembagian data train/test"):
    st.write(
        f"Halaman ini membagi ulang dataset dengan `test_size={TEST_SIZE}` dan "
        f"`random_state={RANDOM_STATE}` untuk menghitung metrik evaluasi. "
        "Jika notebook aslinya memakai parameter split yang berbeda, angka di "
        "halaman ini bisa sedikit berbeda dari hasil di notebook."
    )


@st.cache_data
def load_dataset(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


df = load_dataset(DATASET_PATH)

if df is None:
    st.error(f"Dataset tidak ditemukan di `{DATASET_PATH}`.")
    st.stop()

models_ready = available_models()
if not models_ready:
    st.error("Belum ada model (.pkl) di folder `models/`.")
    st.stop()

try:
    # 1. Encode awal pakai fungsi bawaan project kamu
    df_encoded = encode_dataframe(df)
    X, y = get_features_target(df_encoded)
    
    # 2. Sapu bersih sisa data teks yang belum ter-cover encode_dataframe
    le = LabelEncoder()
    # Cari kolom di X yang tipe datanya masih teks/object
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = le.fit_transform(X[col].astype(str))
        
except ValueError as e:
    st.error(str(e))
    st.stop()

# X sekarang sudah 100% angka, aman untuk di-split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

model_choice = st.selectbox("Pilih Model untuk Dievaluasi", models_ready)

model = load_model(model_choice)
y_pred = model.predict(X_test)

# ----------------------------------------------------------------------
# METRIK UTAMA
# ----------------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

st.subheader(f"📊 Ringkasan Metrik — {model_choice}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{acc:.4f}")
m2.metric("Precision", f"{prec:.4f}")
m3.metric("Recall", f"{rec:.4f}")
m4.metric("F1-Score", f"{f1:.4f}")

st.divider()

# ----------------------------------------------------------------------
# CLASSIFICATION REPORT
# ----------------------------------------------------------------------
st.subheader("📋 Classification Report")
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()
st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

st.divider()

# ----------------------------------------------------------------------
# CONFUSION MATRIX
# ----------------------------------------------------------------------
st.subheader("🔲 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.8,
    linecolor="white",
    cbar=True,
    xticklabels=["Tidak Lulus", "Lulus"],
    yticklabels=["Tidak Lulus", "Lulus"],
    annot_kws={"size": 12},
    ax=ax,
)
ax.set_title(f"Confusion Matrix — {model_choice}", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
st.pyplot(fig)

st.divider()

# ----------------------------------------------------------------------
# FEATURE IMPORTANCE
# ----------------------------------------------------------------------
if hasattr(model, "feature_importances_"):
    st.subheader(f"🌟 Feature Importance — {model_choice}")

    importance_df = pd.DataFrame({
        "Fitur": FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Importance", y="Fitur", data=importance_df, ax=ax2, color="#4C72B0")
    ax2.set_title(f"Feature Importance {model_choice}")
    st.pyplot(fig2)
else:
    st.info(f"Model {model_choice} tidak menyediakan feature_importances_ (contoh: Logistic Regression).")

    if hasattr(model, "coef_"):
        st.subheader("📐 Koefisien Logistic Regression")
        coef_df = pd.DataFrame({
            "Fitur": FEATURES,
            "Koefisien": model.coef_[0],
        }).sort_values(by="Koefisien", key=abs, ascending=False)
        st.dataframe(coef_df, use_container_width=True)