"""
pages/3_Prediksi.py
Halaman form input data mahasiswa baru untuk diprediksi status kelulusannya.
"""

import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import (  # noqa: E402
    encode_single_input,
    PEKERJAAN_OPTIONS,
    KEHADIRAN_OPTIONS,
)
from utils.load_model import available_models, load_model  # noqa: E402

st.set_page_config(page_title="Prediksi", page_icon="🔮", layout="wide")

st.title("🔮 Prediksi Status Kelulusan Mahasiswa")
st.caption("Isi data akademik mahasiswa di bawah, lalu klik **Prediksi**.")

models_ready = available_models()

if not models_ready:
    st.error(
        "Belum ada model (.pkl) yang ditemukan di folder `models/`. "
        "Tambahkan minimal salah satu dari: logistic_regression.pkl, "
        "random_forest.pkl, xgboost.pkl."
    )
    st.stop()

model_choice = st.selectbox("Pilih Model", models_ready)

st.divider()

with st.form("form_prediksi"):
    col1, col2, col3 = st.columns(3)

    with col1:
        ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
        mk_tidak_lulus = st.number_input(
            "Mata Kuliah Tidak Lulus", min_value=0, max_value=50, value=0, step=1
        )
        cuti_akademik = st.number_input(
            "Jumlah Cuti Akademik", min_value=0, max_value=10, value=0, step=1
        )

    with col2:
        pekerjaan = st.selectbox("Pekerjaan Sambil Kuliah", PEKERJAAN_OPTIONS)
        jumlah_semester = st.number_input(
            "Jumlah Semester", min_value=1, max_value=20, value=8, step=1
        )
        kehadiran = st.selectbox("Kategori Kehadiran", KEHADIRAN_OPTIONS)

    with col3:
        ips_rata = st.number_input("IPS Rata-rata", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
        ips_akhir = st.number_input(
            "IPS Semester Akhir", min_value=0.0, max_value=4.0, value=3.0, step=0.01
        )
        ips_tren = st.number_input(
            "IPS Tren (perubahan IPS dari semester sebelumnya)",
            min_value=-4.0,
            max_value=4.0,
            value=0.0,
            step=0.01,
        )

    submitted = st.form_submit_button("🔍 Prediksi", use_container_width=True)

if submitted:
    input_dict = {
        "IPK": ipk,
        "Mata Kuliah Tidak Lulus": mk_tidak_lulus,
        "Jumlah Cuti Akademik": cuti_akademik,
        "Pekerjaan Sambil Kuliah": pekerjaan,
        "Jumlah Semester": jumlah_semester,
        "IPS Rata-rata": ips_rata,
        "IPS Semester Akhir": ips_akhir,
        "IPS Tren": ips_tren,
        "Kategori Kehadiran": kehadiran,
    }

    try:
        X_input = encode_single_input(input_dict)
        model = load_model(model_choice)

        pred = model.predict(X_input)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]

        st.divider()
        st.subheader("📢 Hasil Prediksi")

        if pred == 1:
            st.success("✅ Mahasiswa diprediksi **LULUS**")
        else:
            st.error("⚠️ Mahasiswa diprediksi **TIDAK LULUS**")

        if proba is not None:
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("Probabilitas Tidak Lulus", f"{proba[0]*100:.1f}%")
            p_col2.metric("Probabilitas Lulus", f"{proba[1]*100:.1f}%")

        with st.expander("Lihat data input yang diproses model"):
            st.dataframe(X_input, use_container_width=True)

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
