# 🚀 Research Tools Data Science - K-Means Clustering

Aplikasi Web Interaktif & Framework Modular Data Science untuk Klasterisasi Data Pajak Kendaraan Bermotor (PKB) menggunakan Algoritma **K-Means Clustering** dan **Evaluasi Silhouette Score** berdasarkan 9 Tahapan Metodologi Penelitian Data Science.

---

## 👥 Tim Pengembang
- **Peneliti utama**: Kang Yayat
- **Senior Data Scientist**: Mang Ewok

---

## 📊 Ringkasan Dataset Penelitian
- **Populasi Dataset**: Data Pajak Kendaraan Bermotor (PKB) BAPENDA
- **Ukuran Presisi**: **451.104 Baris Transaksi x 101 Kolom/Atribut** (208.55 MB)
- **Format Dataset**: Excel (`.xlsx` - `Sheet1`)

---

## 📌 Alur 9 Tahapan Penelitian Data Science

1. 🛠️ **Tahap 1: Environment & System Check** (Verifikasi 10 Poin Spesifikasi Lingkungan Penelitian Python, CPU, RAM, GPU, OS, Library)
2. 📥 **Tahap 2: Pengumpulan Data & Profiling 2.1 s/d 2.12** (Calamine Engine Loading, Backup Parquet Timestamp, Optimasi RAM 77.71%, Tabel 5 Kolom Tipe Data, Univariate 2.8, Bivariate 2.9, Multi-Year Time Series 2.10, & Cek Kualitas Data 2.11/2.12)
3. 🎯 **Tahap 3: Seleksi Data 3.1 s/d 3.4** (Reduksi Atribut Terselek 17 Kolom Utama & Korelasi Spearman)
4. 🧹 **Tahap 4: Data Preprocessing 4.1 s/d 4.8** (Cleaning, Handling Missing, Validasi Duplikasi, Deteksi Outlier IQR, Validasi Tanggal)
5. ⚙️ **Tahap 5: Data Transformasi & Normalisasi 5.1 s/d 5.10** (Feature Engineering UsiaKendaraan, RiwayatPembayaran, DurasiTunggakan, Target Encoding Skor Kepatuhan, Log1p, StandardScaler, & Skewness Analysis)
6. 🤖 **Tahap 6: Penambangan Data K-Means** (Sampling 30%, Elbow Method WCSS, Candidate Models K=2..5)
7. ⭐ **Tahap 7: Evaluasi Silhouette Score & PCA** (Silhouette Score Max K=5 = 0.744167 Strong Cluster, 2D & 3D PCA Scatter Plot, Fit 100% Data)
8. 🔍 **Tahap 8: Interpretasi Klaster & Profiling 8.1 s/d 8.8** (Centroid Skala Asli, Reverse Mapping Tabulasi Silang Klaster 0..4, Dominan Kategori, & Analisis Wilayah)
9. 🏆 **Tahap 9: Kesimpulan & Ekspor Multi-Format** (Ringkasan Hasil Penelitian, Ekspor .parquet, .csv, .pkl, & Verifikasi Load .pkl)

---

## ⚡ Instalasi & Cara Penggunaan

### 1. Clone Repository & Install Dependencies
```powershell
git clone https://github.com/ayat7779/K-Means-Clustering.git
cd K-Means-Clustering
pip install -r requirements.txt
```

### 2. Jalankan Web Dashboard Streamlit
```powershell
python -m streamlit run app.py
```

### 3. Jalankan Automated 9-Stage Pipeline (CLI)
```powershell
python run_pipeline.py
```

---

## 📄 Lisensi & Laporan Penelitian
Laporan lengkap 9 tahapan penelitian dapat dilihat pada file [laporan_app.md](laporan_app.md).
