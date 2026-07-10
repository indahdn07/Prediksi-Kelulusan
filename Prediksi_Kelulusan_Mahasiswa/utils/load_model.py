"""
utils/load_model.py

Fungsi untuk memuat model Machine Learning dari folder models/,
dengan caching supaya tidak reload berulang kali tiap interaksi Streamlit.
"""

import os
import pickle
import streamlit as st
from xgboost import XGBClassifier  # Tambahan import khusus XGBoost

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.json",  # Ubah dari .pkl jadi .json
}


def get_model_path(model_name: str) -> str:
    filename = MODEL_FILES.get(model_name)
    if filename is None:
        raise ValueError(f"Model '{model_name}' tidak dikenali.")
    return os.path.join(MODELS_DIR, filename)


def model_available(model_name: str) -> bool:
    """Cek apakah file model tersedia di folder models/."""
    return os.path.exists(get_model_path(model_name))


def available_models() -> list:
    """Daftar nama model yang file-nya sudah ada."""
    return [name for name in MODEL_FILES if model_available(name)]


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """
    Load model. Hasil di-cache oleh Streamlit (st.cache_resource) 
    supaya model hanya dibaca sekali per sesi.
    """
    path = get_model_path(model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File model tidak ditemukan: {path}\n"
            f"Pastikan '{MODEL_FILES[model_name]}' sudah diletakkan di folder models/."
        )
    
    # Penanganan khusus untuk model XGBoost (memakai .json)
    if model_name == "XGBoost":
        model = XGBClassifier()
        model.load_model(path)
        return model
        
    # Untuk Logistic Regression dan Random Forest (memakai .pkl)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model