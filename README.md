# HMTC Fine-Grained Emotion Classification Pipeline 🚀

Repository ini memuat *pipeline* klasifikasi emosi tingkat spesifik (*fine-grained*) berarsitektur **Hierarchical Multi-label Text Classification (HMTC)**, didesain khusus untuk Teks Bahasa Indonesia pada domain Media Sosial. Pipeline ini mendukung eksperimen sistematis yang terkomputasi langsung ke rekam jejak model (MLFlow) untuk keperluan Skripsi.

## 🌟 Fitur Utama
1. **Dukungan Multi-Arsitektur 10 Model**: Mencakup Tradisional (LR, NB, SVM), Deep Learning (Bi-LSTM, CNN), dan Pretrained Language Models (IndoBERT, XLM-R, mmBERT).
2. **Evaluasi HMTC Ekstensif**: Implementasi native atas *Hierarchical Precision/Recall/F1*, *Hamming Loss*, dan *Exact Match Ratio (Subset Accuracy)*.
3. **Analisis Manual (Error Analysis)**: Secara otomatis mengeluarkan prediksi model vs *Ground Truth* ke format CSV yang siap dibaca oleh asisten manusia.
4. **Validasi Ablasi Metodologis**: Selaras dengan *best-practice* jurnal akademik; mendukung penyesuaian ablasi *N-gram (Unigram, Bigram, Trigram)* dan ekstraksi perbandingan *Learning Rates Parameter*.

---

## 🛠 Instalasi dan Konfigurasi

Anda dapat menjalankan *pipeline* ini melalui dua cara: menggunakan **Docker** (Direkomendasikan untuk stabilitas dan komputasi MLOps terotomasi) atau melakukan instalasi **Lokal (Python Virtual Environment)**.

### a. Menjalankan via Docker (Cara Tepat & Cepat)
Cukup pastikan Docker telah berjalan di mesin lokal atau server Anda.
```bash
# Lakukan automasi penyiapan MLFlow Server + Pipeline Trainer secara bersamaan
docker-compose up --build
```
> *(Catatan: Pendekatan ini akan mengunci terminal dengan proses berantai. Akses MLFlow dapat dibuka pada tautan localhost yang keluar dari terminal docker)*

### b. Menjalankan via Lokal (Python Asli)
Jika Anda menargetkan modifikasi ekstensif langsung pada kode dasar atau sudah menggunakan Visual Studio Code Python IDE:
```bash
# 1. Pasang paket dan module yang dibutuhkan
pip install -r requirements.txt loguru

# 2. Opsional: Instal PyTorch varian CUDA untuk percepatan GPU NVIDIA
# (Sesuaikan versi pip install pytorch dari situs remi)

# 3. Jalankan script utamanya (Baca Langkah 3 di bawah)
```

---

## 📂 Panduan Standar Data JSON
Sistem ini mengambil data masuk (input) secara terpusat melalui fail berekstensi `.json` yang wajib ditempatkan pada `data/dataset.json`.
Struktur internal data (daftar dari kamus data) minimal harus memuat label (*keys*) di bawah ini (yang dibentuk dari re-anotasi korpus skripsi awal):
* `text`: Teks media sosial Indonesia murni.
* `new_label_basic` atau `label_basic`: Label induk taksonomi (Contoh: "Love", "Sadness").
* `new_label_fine_grained` atau `label_finegrained`: Label ranting terdalam (Contoh: "Nonsexual desire").
* `splitting`: Alur pembagian data statis (*"train", "val", "test"*).

Data Anda akan ditarik oleh sistem (via `src/data_loader.py`), di mana model akan mencerna label secara dinamis tanpa batas!

---

## 🚀 Tutorial Penggunaan Pipeline Eksekusi

Titik akses eksklusif repositori ini berada di fail `run_pipeline.py`. Anda dapat mengeksekusi arsitektur secara modular:

### 1. Menjalankan Seluruh Model Secara Berantai
```bash
python run_pipeline.py --data_path ./data/dataset.json --run all
```

### 2. Menjalankan Pipeline Khusus Secara Terisolasi
Anda juga dapat mengeksekusi pipeline spesifik untuk memisahkan fokus ekstensif Anda apabila waktu uji coba memakan CPU/GPU yang signifikan:

**A. Pipeline Model Tradisional ML (TF-IDF & Studi Ablasi N-gram):**
```bash
python run_pipeline.py --data_path ./data/dataset.json --run traditional
```
*Menguji LogisticRegression, NaiveBayes (Laplace Scaling), dan SVM (RBF).*

**B. Pipeline Deep Learning (Pytorch):**
```bash
python run_pipeline.py --data_path ./data/dataset.json --run dl
```
*Melakukan ekstraksi ablasi matriks terhadap Bi-LSTM dan CNN (Sentence-level).*

**C. Pipeline Transformers (Hugging Face HPO Target):**
```bash
python run_pipeline.py --data_path ./data/dataset.json --run transformers
```
*Auto-grid terhadap Learning Rates [2e-5, 3e-5, 4e-5, 5e-5] untuk model sekelas indoBERT.*

---

## 📊 Integrasi Pelaporan (*MLFlow Reporting*)
Saat skrip Python di atas berjalan, maka direktori baru bertajuk `mlruns` (/atau `mlflow_data`) akan tercipta untuk menampung riwayat perhitungan parameter berlipat-lipat yang secara *real-time* tercatat oleh eksperimen Anda.

Untuk melihat perbandingan *macro-F1* seluruh model dalam antarmuka web (UI) grafis yang menawan dan interaktif:
```bash
# Jalankan di terminal kedua:
mlflow ui
```
Buka tautan [http://localhost:5000](http://localhost:5000) pada peramban web (*browser*).

---

## 📚 Audit Model Manual (Manual Analysis)
Tesis akademis berstandar tinggi senantiasa memperdebatkan tidak hanya kalkulasi angka metrik semata melainkan juga realita luaran (*output*) klasifikasi (*Error Analysis*). 
Program ini akan secara otomatis meleburnya di dalam tabung output CSV:
1. Temukan direktori pasca-eksekusi di `analysis/`.
2. Buka fail yang berhilir misal: `manual_analysis_SVM_Trigram.csv`.
3. Gunakan *Microsoft Excel / CSV Reader* untuk mempelajari mengapa ada cuitan netizen yang status akurasinya memaparkan pesan *"Salah Total"* vs *"Sebagian Benar"*. 

Selamat menulis dan bereksperimen! Moga sistem analitik komprehensif ini mampu menjadi pondasi cadas penyelesaian Studi Tugas Akhir Anda! 🎓
