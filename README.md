# 🏥 Sistem Skrining Kelainan Suara Jantung (PCG)

**Capstone Project - Machine Learning (Triple Ensemble Architecture)**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green.svg)](https://xgboost.readthedocs.io/)

Proyek ini adalah sistem kecerdasan buatan berbasis _Machine Learning_ untuk mendeteksi kelainan detak jantung (seperti murmur) melalui rekaman suara _Phonocardiogram_ (PCG). Dikembangkan untuk menjawab tantangan ketersediaan Kardiolog di faskes tingkat pertama.

Model ini dilatih menggunakan dataset internasional dari kompetisi **PhysioNet/CinC Challenge 2016**, dan berhasil melampaui skor Juara 1 Dunia kompetisi tersebut.

---

## 📊 Performa Model (vs Juara 1 Dunia)

Metrik utama yang digunakan adalah **MAcc (Modified Accuracy)**, yaitu rata-rata dari _Sensitivity_ dan _Specificity_, untuk mengatasi ketidakseimbangan kelas (_Class Imbalance_ 80:20).

| Metrik          | Skor Model Kami (Triple Ensemble) | Skor Juara 1 (Potes et al. 2016) |
| :-------------- | :-------------------------------: | :------------------------------: |
| **MAcc**        |           **88.03%** 🏆           |              86.02%              |
| **Sensitivity** |              90.23%               |                -                 |
| **Specificity** |              85.83%               |                -                 |

_(Catatan: Model kami dievaluasi menggunakan 3.240 data asli 100% tanpa augmentasi sintetis, dengan threshold optimal 0.19 berdasarkan protokol PhysioNet)._

---

## 🧠 Arsitektur Triple Ensemble

Kami menggunakan 3 "Otak" buatan yang mengamati data dari sudut pandang berbeda:

1. **Otak 1 (CRNN - EfficientNetB0 + BiLSTM):** Mengobservasi pola visual dari gambar spektogram (_Mel-Spectrogram 2D_). Bobot: 15%.
2. **Otak 2 (1D-CNN):** Mendengarkan langsung pola waktu berurutan dari sinyal gelombang suara mentah. Bobot: 5%.
3. **Otak 3 (XGBoost):** Menganalisis 33 fitur statistik matematis (MFCC, _Shannon Energy_, _Spectral Rolloff_ dll). Bobot: 80%.

Ketiga prediksi digabungkan menggunakan teknik _Soft-Voting_ dengan bobot yang dioptimasi via _Grid Search_.

---

## 🌐 Fitur Aplikasi Web (Streamlit)

Aplikasi web interaktif ini dibangun menggunakan Streamlit dan didesain untuk memiliki 5 fitur/halaman utama:

1. **Dashboard EDA:** Visualisasi interaktif dari analisis eksplorasi data (distribusi kelas Normal/Abnormal, panjang rekaman, perbandingan waveform dan spectrogram).
2. **Model Demo:** _Interface_ bagi tenaga medis untuk mengunggah file data audio (`.wav`) suara jantung dan mendapatkan hasil prediksi (_Screening_) secara langsung dari model AI terbaik.
3. **Evaluasi Model:** Tampilan visual untuk metrik evaluasi model seperti kurva ROC-AUC, Confusion Matrix, dan perbandingan performa.
4. **Interpretasi Hasil:** Penjelasan _white-box_ model menggunakan SHAP (SHapley Additive exPlanations) untuk mengetahui mengapa AI memberikan prediksi tertentu beserta insights klinisnya.
5. **Dokumentasi:** Penjelasan lengkap tentang sumber dataset PhysioNet 2016, metodologi arsitektur _Triple Ensemble_, dan panduan penggunaan aplikasi.

---

## 📂 Struktur Repositori (Struktur Standar + Current State)

Sesuai dengan standar industri dan arahan akademik (dosen), proyek ini disusun dengan struktur _core_ yang rapi. Namun, terdapat juga penambahan folder dataset fisik (`training-*` & `validation`) dari PhysioNet dan beberapa file ekstra untuk pengujian.

Berikut adalah rincian beserta penjelasannya:

```text
kardiolog.ai/
│
# ── CORE STRUCTURE (Standar Akademik/Industri) ──
│
├── 📂 .streamlit/                    # Konfigurasi tampilan Streamlit
│   └── config.toml                   # Tema warna, layout, dan setting UI aplikasi
│
├── 📂 app/                           # Source code utama antarmuka pengguna (UI)
│   └── app.py                        # Script Dashboard Web Streamlit (5 halaman:
│                                     #   Dashboard, Prediksi AI, Evaluasi Model,
│                                     #   Interpretasi SHAP, Dokumentasi)
│
├── 📂 data/                          # Folder untuk data tabular & ekstraksi fitur
│   └── raw/                          # Data mentah (file .wav hasil rekaman jantung)
│
├── 📂 models/                        # Penyimpanan file model Pre-trained
│   ├── crnn_model.h5                 # Model Deep Learning #1 — CRNN (EfficientNetB0 + BiLSTM)
│   │                                 #   Input: Mel-Spectrogram 2D (gambar visual sinyal)
│   ├── cnn1d_model.h5                # Model Deep Learning #2 — 1D-CNN
│   │                                 #   Input: Raw Waveform (sinyal audio mentah)
│   ├── xgboost_model.json            # Model Machine Learning #3 — XGBoost
│   │                                 #   Input: 33 Fitur Statistik (MFCC, Shannon Energy, dll)
│   └── scaler.pkl                    # Object StandardScaler untuk normalisasi fitur statistik
│
├── 📂 notebooks/                     # Notebook riset dan eksplorasi data (Jupyter)
│   ├── 01_eda.ipynb                  # Tahap 1: Eksplorasi & visualisasi data (EDA)
│   ├── 02_modeling.ipynb             # Tahap 2: Pelatihan dan evaluasi ketiga model
│   ├── 03_interpretation.ipynb       # Tahap 3: Analisis interpretabilitas model (SHAP)
│   └── Triple Ensemble PCG.IPY       # Notebook utama lengkap — gabungan seluruh alur riset
│                                     #   dari EDA hingga deployment Triple Ensemble
│
├── 📂 reports/                       # Dokumentasi dan pelaporan akademik
│   └── Laporan_Riset.md              # Laporan teknis mendalam dalam format Markdown
│
├── 📂 src/                           # Source code pembantu (Utility & Pipeline)
│   ├── data_preprocessing.py         # Script ekstraksi fitur audio
│   │                                 #   (Butterworth Filter, Librosa, MFCC, Spectral features)
│   ├── train_model.py                # Script pipeline pelatihan model (CRNN, 1D-CNN, XGBoost)
│   ├── evaluate_model.py             # Script evaluasi model (MAcc, Sensitivity, Specificity, AUC)
│   └── utils.py                      # Script fungsi umum yang dipakai bersama
│

# ── ROOT FILES ──
│
├── 📄 convert_model.py               # Script konversi format model (mis: .pkl → .json untuk XGBoost)
├── 📄 a0006 (ABNORMAl).wav           # File audio contoh — Pasien SAKIT (untuk uji cepat di aplikasi)
├── 📄 a0007 (NORMAL).wav             # File audio contoh — Pasien SEHAT (untuk uji cepat di aplikasi)
├── 📄 .gitignore                     # Daftar file/folder yang diabaikan oleh Git
│                                     #   (mis: folder dataset besar tidak di-push ke GitHub)
├── 📄 requirements.txt               # Daftar seluruh library Python yang dibutuhkan:
│                                     #   streamlit, librosa, tensorflow, xgboost, plotly, dll
└── 📄 README.md                      # File dokumentasi utama yang sedang Anda baca ini
```

---

## 🚀 Cara Menjalankan Aplikasi (Streamlit)

1. **Clone Repositori ini:**

   ```bash
   git clone https://github.com/fauzixhuman/kardiolog.ai.git
   cd kardiolog.ai
   ```

2. **Install Library:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Streamlit:**
   ```bash
   streamlit run app/app.py
   ```

Aplikasi akan otomatis terbuka di browser (`localhost:8501`). Anda bisa mengunggah file `.wav` dari folder `data/raw/` untuk melihat AI melakukan prediksi secara _real-time_.

---

## 👨‍🏫 Panduan Pengujian

Kami telah menyediakan **2 file sampel** langsung di repositori GitHub ini untuk memudahkan pengujian:

1. 🔴 [`a0006 (ABNORMAl).wav`](<a0006%20(ABNORMAl).wav>) — Sampel rekaman pasien yang terdeteksi memiliki kelainan jantung.
2. 🟢 [`a0007 (NORMAL).wav`](<a0007%20(NORMAL).wav>) — Sampel rekaman pasien dengan suara jantung sehat/normal.

**Cara Melakukan Pengujian:**

1. Klik salah satu tautan file `.wav` di atas (akan membuka halaman file di GitHub).
2. Klik tombol **Download raw file** (ikon 📥 di kanan atas kotak kode) untuk menyimpan file `.wav` ke laptop Anda.
3. Buka link website aplikasi Kardiolog.ai yang sudah di-_deploy_.
4. Masuk ke menu **"✨ Prediksi AI"**.
5. _Drag & Drop_ (atau klik _Browse_) file `.wav` yang baru saja Anda download ke dalam kotak unggahan.
6. Lihat bagaimana model Triple Ensemble AI Saya melakukan analisis dan memberikan diagnosis secara _real-time_

---

## 🔬 Interpretasi AI (SHAP)

Model ini tidak bertindak sebagai _Black-Box_. Menggunakan library SHAP, AI mengkonfirmasi bahwa **Shannon Energy** dan **MFCC** adalah fitur paling menentukan. Ini terbukti selaras dengan literatur medis di mana _murmur_ jantung akan merusak keteraturan (_entropy/energy_) dari pola suara degup normal (S1-S2).
