# 🎓 Prediksi Kelulusan Mahasiswa

Aplikasi Streamlit untuk memprediksi status kelulusan mahasiswa (**Lulus** / **Tidak Lulus**)
menggunakan model Machine Learning: Logistic Regression, Random Forest, dan XGBoost.

## 📁 Struktur Project

```
Prediksi_Kelulusan_Mahasiswa/
├── app.py                     # Halaman utama (Home)
├── requirements.txt
├── README.md
│
├── models/                    # Simpan model hasil training (.pkl) di sini
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── dataset/
│   └── dataset_kelulusan_mahasiswa.csv
│
├── assets/
│   ├── logo.png
│   └── banner.png
│
├── pages/
│   ├── 1_Dataset.py            # Eksplorasi data mentah
│   ├── 2_Visualisasi.py        # Korelasi & distribusi fitur
│   ├── 3_Prediksi.py           # Form input & prediksi
│   ├── 4_Evaluasi.py           # Metrik evaluasi model
│   └── 5_Tentang.py            # Info project
│
└── utils/
    ├── load_model.py           # Load model .pkl (dengan cache)
    └── preprocessing.py        # Encoding fitur & persiapan data
```

## 🚀 Cara Menjalankan

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Pastikan file berikut sudah ada di tempatnya:
   - `dataset/dataset_kelulusan_mahasiswa.csv`
   - `models/logistic_regression.pkl`, `models/random_forest.pkl`, `models/xgboost.pkl`
     (minimal salah satu)

3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

## ⚠️ Catatan Penting (Perlu Disesuaikan)

Beberapa hal berikut dibuat sebagai **asumsi** karena notebook Colab asli tidak bisa
diakses langsung saat pembuatan project ini. Sesuaikan dengan implementasi aslimu:

- **`utils/preprocessing.py`** — mapping encoding untuk kolom kategorikal
  `Pekerjaan Sambil Kuliah` (Ya/Tidak) dan `Kategori Kehadiran` (Rendah/Sedang/Tinggi).
  Pastikan urutan dan nilai mapping ini sama persis dengan yang dipakai saat training model,
  supaya hasil prediksi konsisten.
- **`pages/4_Evaluasi.py`** — parameter `test_size=0.2` dan `random_state=42` untuk
  split data train/test adalah perkiraan (berdasarkan jumlah data test = 20.000 dari
  total 100.000 baris pada notebook). Sesuaikan jika notebook asli memakai nilai lain.
- Kolom `IPS Tren` diasumsikan sebagai selisih/tren IPS antar semester — sesuaikan
  definisi & rentang nilainya jika berbeda di notebook asli.

## 🧠 Fitur yang Digunakan Model

`IPK`, `Mata Kuliah Tidak Lulus`, `Jumlah Cuti Akademik`, `Pekerjaan Sambil Kuliah`,
`Jumlah Semester`, `IPS Rata-rata`, `IPS Semester Akhir`, `IPS Tren`, `Kategori Kehadiran`

Target: `Status Kelulusan` (0 = Tidak Lulus, 1 = Lulus)
