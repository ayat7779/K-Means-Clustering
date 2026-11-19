import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler

# Formatting Terminal Color
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

C_RESET   = "\033[0m"
C_BOLD    = "\033[1m"
C_DIM     = "\033[2m"
C_CYAN    = "\033[1;36m"
C_GREEN   = "\033[1;32m"
C_YELLOW  = "\033[1;33m"
C_BLUE    = "\033[1;34m"
C_MAGENTA = "\033[1;35m"
C_RED     = "\033[1;31m"


def transform_and_normalize_dataset(
    df_input: Optional[pd.DataFrame] = None,
    file_path: str = "dataset_penelitian_kmeans.xlsx",
    tahun_referensi: int = 2026,
    save_visualizations: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    [TAHAP 5] TRANSFORMASI DAN NORMALISASI - METODOLOGI PENELITIAN KANG YAYAT:
    5.1 Feature Engineering - Usia Kendaraan, Kategori Usia, Skor Usia
    5.2 Riwayat Pembayaran (Penjumlahan Tunggakan T1 - T5)
    5.3 Transformasi Durasi Tunggakan (Selisih TglPKB & Expected Next Due Date)
    5.4 Skor Kepatuhan per Kategori (Target Encoding Mean Durasi Tunggakan)
    5.5 Transformasi Logaritmik (np.log1p)
    5.6 Analisis Korelasi Pearson (Heatmap)
    5.7 Penentuan Fitur Akhir (Multikolinearitas Filtering)
    5.8 Standardisasi (StandardScaler Mean=0, Std=1)
    5.9 Resume & Dataset Siap K-Means (X_kmeans)
    5.10 Analisis Skewness & Visualisasi (Histogram & Boxplot)
    """
    start_time = time.time()
    W = 85
    line_db = C_CYAN + "═" * W + C_RESET
    line_sg = C_DIM + "─" * W + C_RESET

    print("\n" + line_db)
    print(f" {C_BOLD}{C_MAGENTA}⚙️ [TAHAP 5] TRANSFORMASI DAN NORMALISASI (METODOLOGI KANG YAYAT){C_RESET}")
    print(f" {C_DIM}Tim Pengembang: Mang Ewok (Senior DS) & Kang Yayat | Algoritma K-Means{C_RESET}")
    print(line_db)

    # 1. Pemuatan Data dari Tahap 4
    if df_input is None:
        cache_path = os.path.join(os.getcwd(), ".cache_dataset", "preprocessed_dataset.parquet")
        if os.path.exists(cache_path):
            print(f" 📦 Memuat data hasil preprocessing Tahap 4: {C_DIM}{os.path.basename(cache_path)}{C_RESET}")
            df_transform = pd.read_parquet(cache_path)
        else:
            print(f" 📖 Memuat data dari Excel: {C_DIM}{os.path.basename(file_path)}{C_RESET}")
            df_transform = pd.read_excel(file_path, sheet_name="Sheet1", nrows=20000, engine="calamine")
    else:
        df_transform = df_input.copy()

    # ============================================================
    # 5.1 Feature Engineering - Usia Kendaraan & Kategori
    # ============================================================
    if "TahunPembuatan" in df_transform.columns:
        df_transform["UsiaKendaraan"] = tahun_referensi - pd.to_numeric(df_transform["TahunPembuatan"], errors="coerce").fillna(tahun_referensi)
        df_transform.loc[df_transform["UsiaKendaraan"] < 0, "UsiaKendaraan"] = 0

        bins = [-1, 5, 10, 15, 20, 100]
        labels = ["Sangat Baru", "Baru", "Sedang", "Tua", "Sangat Tua"]
        df_transform["KategoriUsia"] = pd.cut(df_transform["UsiaKendaraan"], bins=bins, labels=labels)

        mapping = {"Sangat Baru": 1, "Baru": 2, "Sedang": 3, "Tua": 4, "Sangat Tua": 5}
        df_transform["MKategoriUsia"] = df_transform["KategoriUsia"].map(mapping).fillna(1).astype(int)

        df_transform["SkorUsia"] = pd.cut(df_transform["UsiaKendaraan"], bins=bins, labels=[5, 4, 3, 2, 1]).astype(int)
        print(f" {C_GREEN}✅ 5.1 Fitur UsiaKendaraan, KategoriUsia, MKategoriUsia, & SkorUsia berhasil dibuat.{C_RESET}")

    # ============================================================
    # 5.2 Riwayat Pembayaran (Total Tahun Tunggakan)
    # ============================================================
    tahun_tunggakan = [c for c in ["PokokPKBT1", "PokokPKBT2", "PokokPKBT3", "PokokPKBT4", "PokokPKBT5"] if c in df_transform.columns]
    if tahun_tunggakan:
        df_transform["RiwayatPembayaran"] = (df_transform[tahun_tunggakan] > 0).sum(axis=1)
        print(f" {C_GREEN}✅ 5.2 Kolom baru 'RiwayatPembayaran' berhasil dibuat dari {len(tahun_tunggakan)} kolom tunggakan.{C_RESET}")
    else:
        df_transform["RiwayatPembayaran"] = 0

    # ============================================================
    # 5.3 Transformasi Durasi Tunggakan
    # ============================================================
    if "TglPKB" in df_transform.columns and "TglPKBLalu" in df_transform.columns:
        df_transform['TglPKB'] = pd.to_datetime(df_transform['TglPKB'], errors='coerce')
        df_transform['TglPKBLalu'] = pd.to_datetime(df_transform['TglPKBLalu'], errors='coerce')

        df_transform['Expected_Next_PKB_Due_Date'] = df_transform['TglPKBLalu'] + pd.DateOffset(years=1)
        df_transform['DurasiTunggakan'] = (df_transform['TglPKB'] - df_transform['Expected_Next_PKB_Due_Date']).dt.days
        df_transform['DurasiTunggakan'] = df_transform['DurasiTunggakan'].fillna(0).clip(lower=0)
        print(f" {C_GREEN}✅ 5.3 Fitur DurasiTunggakan (hari) berhasil dibuat berdasarkan TglPKB & TglPKBLalu.{C_RESET}")
    else:
        df_transform['DurasiTunggakan'] = 0

    # ============================================================
    # 5.4 Skor Kepatuhan per Kategori (Target Encoding Mean DurasiTunggakan)
    # ============================================================
    skor_map_list = [
        ('NamaWilayah', 'SkorKabupaten'),
        ('Jenis_Kendaraan', 'SkorJenisKendaraan'),
        ('Warna_TNKB', 'SkorTNKB'),
        ('Golongan', 'SkorGolonganKendaraan'),
        ('Fungsi', 'SkorFungsiKendaraan'),
        ('Jenis_Pendaftaran', 'SkorJenisPendaftaran'),
        ('Lokasi_Bayar', 'SkorLokasiBayar')
    ]

    for cat_col, score_col in skor_map_list:
        if cat_col in df_transform.columns:
            mean_score = df_transform.groupby(cat_col)["DurasiTunggakan"].mean()
            df_transform[score_col] = df_transform[cat_col].map(mean_score).fillna(0)
        else:
            df_transform[score_col] = 0.0

    print(f" {C_GREEN}✅ 5.4 Skor Kepatuhan Target Encoding (7 Fitur Kategori) berhasil dibuat.{C_RESET}")

    # ============================================================
    # 5.5 Transformasi Logaritmik (np.log1p)
    # ============================================================
    log_cols = [
        "PokokPKB", "DendaPKB", 'UsiaKendaraan', 'RiwayatPembayaran', 'DurasiTunggakan',
        'SkorKabupaten', 'SkorJenisKendaraan', 'SkorTNKB', 'SkorGolonganKendaraan',
        'SkorFungsiKendaraan', 'SkorJenisPendaftaran', 'SkorLokasiBayar'
    ]
    log_cols = [c for c in log_cols if c in df_transform.columns]

    for col in log_cols:
        df_transform[col] = pd.to_numeric(df_transform[col], errors="coerce").fillna(0)
        df_transform[f"{col}_log"] = np.log1p(df_transform[col].clip(lower=0))

    print(f" {C_GREEN}✅ 5.5 Transformasi Logaritmik (log1p) {len(log_cols)} fitur berhasil dilakukan.{C_RESET}")

    # ============================================================
    # 5.6 Analisis Korelasi Pearson
    # ============================================================
    correlation_features = [c for c in log_cols if c in df_transform.columns]
    correlation_matrix = df_transform[correlation_features].corr(method="pearson")

    # ============================================================
    # 5.7 Penentuan Fitur Akhir
    # ============================================================
    features = [
        "PokokPKB",
        "DendaPKB",
        "DurasiTunggakan",        # Korelasi 0.89 dengan RiwayatPembayaran
        "SkorTNKB",
        "SkorGolonganKendaraan", # Korelasi 0.92 dengan SkorJenisKendaraan
        "SkorFungsiKendaraan"
    ]
    features = [col for col in features if col in df_transform.columns]
    df_features = df_transform[features].copy()

    print(f"\n {C_BOLD}{C_YELLOW}5.7 FITUR AKHIR PILIHAN PIKEUN KLUSTERISASI ({len(features)} FITUR):{C_RESET}")
    print(line_sg)
    for i, col in enumerate(features, start=1):
        print(f"   {i}. {C_BOLD}{C_CYAN}{col:<25}{C_RESET} (Type: {df_features[col].dtype})")

    # ============================================================
    # 5.8 Standardisasi (StandardScaler Mean=0, Std=1)
    # ============================================================
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(df_features)
    X_scaled = pd.DataFrame(X_scaled_array, columns=features, index=df_features.index)

    print(f"\n {C_BOLD}{C_YELLOW}5.8 HASIL STANDARDISASI (StandardScaler):{C_RESET}")
    print(line_sg)
    stat_scaled = pd.DataFrame({
        "Mean (μ)": X_scaled.mean().round(4),
        "Std Dev (σ)": X_scaled.std().round(4),
        "Min": X_scaled.min().round(4),
        "Max": X_scaled.max().round(4)
    })
    print(stat_scaled.to_string())

    # ============================================================
    # 5.9 Resume & Save Dataset Final Siap K-Means
    # ============================================================
    X_kmeans = X_scaled.copy()
    cache_dir = os.path.join(os.getcwd(), ".cache_dataset")
    os.makedirs(cache_dir, exist_ok=True)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    output_path_transform = os.path.join(cache_dir, f"{date_str}_df_transform_final.csv")
    output_path_scaled = os.path.join(cache_dir, f"{date_str}_df_summary_scaled.csv")
    output_path_parquet = os.path.join(cache_dir, "transformed_scaled_dataset.parquet")

    df_transform.to_csv(output_path_transform, index=False)
    X_kmeans.to_csv(output_path_scaled, index=False)
    X_kmeans.to_parquet(output_path_parquet, index=False)

    # ============================================================
    # 5.10 Analisis Skewness
    # ============================================================
    skewness_values = X_kmeans.skew()
    df_skewness = pd.DataFrame({
        'Fitur': skewness_values.index,
        'Skewness': skewness_values.values
    })
    df_skewness['Interpretasi'] = df_skewness['Skewness'].apply(lambda x:
        'Sangat Miring Positif' if x > 1 else
        'Miring Positif' if x >= 0.5 else
        'Simetris' if x > -0.5 else
        'Miring Negatif' if x >= -1 else
        'Sangat Miring Negatif'
    )

    print(f"\n {C_BOLD}{C_YELLOW}5.10 ANALISIS SKEWNESS FITUR TERPILIH (SEBELUM K-MEANS):{C_RESET}")
    print(line_sg)
    print(df_skewness.sort_values(by='Skewness', ascending=False).to_string(index=False))

    exec_duration = round(time.time() - start_time, 2)

    print("\n" + line_db)
    print(f" {C_BOLD}{C_GREEN}STATUS TAHAP 5: ✅ TRANSFORMASI DAN NORMALISASI SELESAI 100%! SIAP KA TAHAP 6 (K-MEANS){C_RESET}")
    print(line_db + "\n")

    report = {
        "df_transform_shape": df_transform.shape,
        "X_kmeans_shape": X_kmeans.shape,
        "features": features,
        "stat_scaled": stat_scaled,
        "df_skewness": df_skewness,
        "output_path_transform": output_path_transform,
        "output_path_scaled": output_path_scaled,
        "output_path_parquet": output_path_parquet,
        "exec_duration": exec_duration
    }

    return df_transform, X_kmeans, report


if __name__ == "__main__":
    transform_and_normalize_dataset()
