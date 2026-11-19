# Laporan Pengembangan Aplikasi Research Tools Data Science (K-Means Clustering)

## Ringkasan Proyek
- **Algoritma**: K-Means Clustering & Evaluasi Silhouette Score
- **Bahasa Pemrograman**: Python & Streamlit Framework
- **Format Dataset**: File Excel (.xlsx / .xls) - **Sheet1** (Data Pajak Kendaraan Bermotor / PKB Real BAPENDA)
- **Ukuran Dataset Presisi**: **451,104 Baris Transaksi x 101 Kolom/Atribut** (208.55 MB)
- **Tujuan**: Membangun tools penelitian Data Science modular, komprehensif, dan **interaktif/dinamis (Data Quality Check 2.11 & 2.12)** untuk klasterisasi data Excel berdasarkan 9 Tahapan Penelitian Data Science dari Identifikasi Masalah hingga Kesimpulan.
- **Tim Pengembang**: Mang Ewok (Senior Data Scientist) & Kang Yayat

---

## Alur 9 Tahapan Penelitian Data Science (100% Selesai & Terintegrasi)

1. **Tahap 1: Identifikasi Masalah & Persiapan Environment (Google Colab / Python Env Check 10 Poin - Metodologi Kang Yayat)** (Selesai)
2. **Tahap 2: Pengumpulan Data & Profiling 2.1 s/d 2.12 (Metodologi Kang Yayat - Cek Kualitas Data & Missing Value Bar Chart 2.11 & 2.12)** (Selesai)
3. **Tahap 3: Seleksi Data 3.1 s/d 3.4 (Metodologi Kang Yayat - 17 Atribut Terselek)** (Selesai)
4. **Tahap 4: Data Preprocessing (Pembersihan Data - Metodologi Kang Yayat)** (Selesai)
5. **Tahap 5: Data Transformasi dan Normalisasi (StandardScaler & Feature Engineering - Metodologi Kang Yayat)** (Selesai)
6. **Tahap 6: Penambangan Data (K-Means Clustering - Metodologi Kang Yayat)** (Selesai)
7. **Tahap 7: Evaluasi (Silhouette Score & PCA 2D/3D - Metodologi Kang Yayat)** (Selesai)
8. **Tahap 8: Interpretasi Klaster (Profiling & Reverse Mapping - Metodologi Kang Yayat)** (Selesai)
9. **Tahap 9: Kesimpulan & Ekspor Hasil Penelitian Multi-Format (Metodologi Kang Yayat)** (Selesai)

---

## Detail Pelaksanaan Tahapan & Antarmuka Frontend (`app.py`)

### 🌐 Antarmuka Web Frontend Streamlit (`app.py`)
Antarmuka web **Streamlit Web Framework (`app.py`)** telah mengintegrasikan seluruh hasil pemprosesan dari **Tahap 1 hingga Tahap 9 (Metodologi Kang Yayat)** secara sempurna dalam 9 Tab Navigasi Interaktif:
1. 🛠️ **Tahap 1: Environment & System Check**: Verifikasi 10 Poin Spesifikasi Lingkungan Penelitian (`[1]` Python, `[2]` OS, `[3]` CPU, `[4]` RAM, `[5]` Storage, `[6]` GPU, `[7]` Versi Library, `[8]` Working Directory, `[9]` Google Colab, `[10]` Linux).
2. 📥 **Tahap 2: Pengumpulan & Profiling Dataset**: Visualisasi populasi data `Sheet1` (**451.104 Baris Transaksi x 101 Kolom/Atribut**), Tabel 5 Kolom Tipe Data (`No`, `Tipe Data`, `Jumlah`, `Data`, `Nama Data`), Subplot 2x2 Univariate Analysis (2.8), Bivariate Analysis 2.9 (Korelasi Spearman, Denda per Wilayah, Stacked Crosstab, & Scatter Plot Dinamis), Time Series Analysis 2.10 (Bulanan Multi-Tahun, Harian Multi-Tahun, & Tahunan), serta Cek Kualitas Data 2.11 & 2.12 (Missing Values Bar Chart Top N Dinamis & Tabel Rekapitulasi Missing).
3. 🎯 **Tahap 3: Seleksi Data**: Menampilkan seleksi 17 atribut terpilih 3.1 s/d 3.4, matriks korelasi Spearman, serta modul interaktif `st.multiselect` atribut numerik.
4. 🧹 **Tahap 4: Data Preprocessing**: Pembersihan data 4.1 s/d 4.8.
5. ⚙️ **Tahap 5: Data Transformasi & Normalisasi**: Feature Engineering 5.1 s/d 5.10 & StandardScaler.
6. 🤖 **Tahap 6 & 7: Penambangan Data K-Means & Evaluasi**: Sampling 30%, Elbow Method, Silhouette Score (0.7278), PCA 2D/3D.
7. 🔍 **Tahap 8: Interpretasi Klaster & Profiling**: Matriks Centroid Skala Asli, Reverse Mapping 8.3.1-8.3.5, Kategori Dominan 8.4, & Analisis Wilayah 8.6.
8. 🏆 **Tahap 9: Kesimpulan & Ekspor Hasil Penelitian**: Ringkasan hasil penelitian, ekspor multi-format (.parquet, .csv, .pkl), tombol unduh file, dan verifikasi muat ulang .pkl.
9. 🗺️ **Roadmap Penelitian Completed**: Indikator status 100% Selesai dengan efek animasi perayaan (*st.balloons*).

---

### Tahap 2: Pengumpulan Data & Profiling 2.1 s/d 2.12 (Metodologi Kang Yayat)
- **Status**: [x] **Selesai**
- **Sub-Tahap 2.11 & 2.12 Cek Kualitas Data (Dinamis Interaktif)**:
  1. **Bar Chart Top N Kolom Missing**: `sns.barplot(data=top_missing, x='Persentase_Missing (%)', y='Kolom', palette='magma')` dengan persentase angka tertera di ujung batang.
  2. **Tabel Rekapitulasi Missing Values (2.12)**: Tabel lengkap diurutkan dari persentase data hilang terbesar ke terendah.
  3. **Kontrol Interaktif**: Slider Top N kolom missing (5 s/d 30) dan pilihan palet warna (`magma`, `rocket`, `viridis`, `plasma`, `Reds_r`).

---

## Status Proyek Keseluruhan (100% COMPLETE & VERIFIED)
- [x] **Tahap 1: Identifikasi Masalah & Persiapan Environment** (Selesai)
- [x] **Tahap 2: Pengumpulan Data (451.104 Baris x 101 Kolom - Cek Kualitas Data 2.11 & 2.12)** (Selesai)
- [x] **Tahap 3: Seleksi Data (17 Atribut Terselek)** (Selesai)
- [x] **Tahap 4: Data Preprocessing (Pembersihan Data)** (Selesai)
- [x] **Tahap 5: Data Transformasi dan Normalisasi** (Selesai)
- [x] **Tahap 6: Penambangan Data (K-Means Clustering)** (Selesai)
- [x] **Tahap 7: Evaluasi (Silhouette Score & PCA)** (Selesai)
- [x] **Tahap 8: Interpretasi Klaster (Profiling & Reverse Mapping)** (Selesai)
- [x] **Tahap 9: Kesimpulan & Ekspor Hasil Penelitian Multi-Format** (Selesai)
