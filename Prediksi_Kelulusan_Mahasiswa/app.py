# ----------------------------------------------------------------------
# STATUS MODEL
# ----------------------------------------------------------------------
st.subheader("🤖 Status Model")

# Dikembalikan jadi 3 model
models_info = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.json", # <-- sesuaikan, kalau di kamu pakainya .pkl, ganti jadi xgboost.pkl ya
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