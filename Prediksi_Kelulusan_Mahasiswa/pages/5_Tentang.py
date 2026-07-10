"""
pages/5_Tentang.py
Halaman informasi tentang project.
"""

import streamlit as st

st.set_page_config(page_title="Tentang", page_icon="ℹ️", layout="wide")

st.title("ℹ️ Tentang Project")

st.markdown(
    """
    ## 🎓 Prediksi Kelulusan Mahasiswa

    Project ini bertujuan membangun model *Machine Learning* untuk memprediksi
    apakah seorang mahasiswa berpotensi **Lulus** atau **Tidak Lulus**,
    berdasarkan data akademik seperti IPK, IPS per semester, jumlah cuti akademik,
    status pekerjaan sambil kuliah, dan kategori kehadiran.

    ### 🧠 Model yang Digunakan
    - **Logistic Regression** — model linear sebagai baseline
    - **Random Forest** — model ensemble berbasis banyak decision tree
    - **XGBoost** — model gradient boosting untuk performa yang lebih optimal

    ### 🧩 Fitur yang Digunakan
    | Fitur | Keterangan |
    |---|---|
    | IPK | Indeks Prestasi Kumulatif mahasiswa |
    | Mata Kuliah Tidak Lulus | Jumlah mata kuliah yang belum lulus |
    | Jumlah Cuti Akademik | Berapa kali mahasiswa mengambil cuti akademik |
    | Pekerjaan Sambil Kuliah | Apakah mahasiswa bekerja sambil kuliah (Ya/Tidak) |
    | Jumlah Semester | Total semester yang telah ditempuh |
    | IPS Rata-rata | Rata-rata Indeks Prestasi Semester |
    | IPS Semester Akhir | IPS pada semester terakhir |
    | IPS Tren | Perubahan/tren IPS antar semester |
    | Kategori Kehadiran | Tingkat kehadiran mahasiswa (Rendah/Sedang/Tinggi) |

    ### ⚙️ Alur Aplikasi
    1. **Dataset** — eksplorasi data mentah
    2. **Visualisasi** — memahami pola & korelasi antar fitur
    3. **Prediksi** — memasukkan data mahasiswa baru untuk diprediksi
    4. **Evaluasi** — melihat performa masing-masing model

    ---
    ⚠️ **Disclaimer:** Hasil prediksi bersifat estimasi berdasarkan pola data historis
    dan tidak dimaksudkan sebagai keputusan akademik final.
    """
)

st.divider()
st.caption("Dibuat dengan ❤️ menggunakan Streamlit • Project Prediksi Kelulusan Mahasiswa")
