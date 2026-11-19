import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List, Optional

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


def select_features_for_kmeans(
    df_input: Optional[pd.DataFrame] = None,
    file_path: str = "dataset_penelitian_kmeans.xlsx",
    nrows: Optional[int] = 20000
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    [TAHAP 3] SELEKSI DATA (DATA SELECTION) - METODOLOGI PENELITIAN KANG YAYAT:
    3.1 Daftar Atribut Hasil Data Selection (15 Atribut Utama)
    3.2 Membentuk Dataset Hasil Seleksi (df_selected)
    3.3 Validasi Dataset Hasil Seleksi (Missing Values & Duplikasi)
    3.4 Menyimpan Dataset Hasil Seleksi (.parquet & .csv timestamped)
    """
    start_time = time.time()
    W = 85
    line_db = C_CYAN + "═" * W + C_RESET
    line_sg = C_DIM + "─" * W + C_RESET

    print("\n" + line_db)
    print(f" {C_BOLD}{C_MAGENTA}🎯 [TAHAP 3] SELEKSI DATA / DATA SELECTION (METODOLOGI KANG YAYAT){C_RESET}")
    print(f" {C_DIM}Tim Pengembang: Mang Ewok (Senior DS) & Kang Yayat | Algoritma K-Means{C_RESET}")
    print(line_db)

    # 1. Pemuatan Data Optimasi ti Tahap 2 / Cache
    cache_dir = os.path.join(os.getcwd(), ".cache_dataset")
    output_dir = os.path.join(os.getcwd(), "Hasil_Penelitian_PKB", "01_Dataset")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if df_input is None:
        raw_cache_path = os.path.join(cache_dir, "raw_dataset_cache.parquet")
        if os.path.exists(raw_cache_path):
            print(f" 📦 Memuat data optimasi ti Tahap 2 Cache: {C_DIM}{os.path.basename(raw_cache_path)}{C_RESET}")
            df_optimasi = pd.read_parquet(raw_cache_path)
            if nrows and len(df_optimasi) > nrows:
                df_optimasi = df_optimasi.iloc[:nrows].copy()
        else:
            print(f" 📖 Memuat data dari Excel: {C_DIM}{os.path.basename(file_path)}{C_RESET}")
            df_optimasi = pd.read_excel(file_path, sheet_name="Sheet1", nrows=nrows, engine="calamine")
    else:
        df_optimasi = df_input.copy()

    # ============================================================
    # [TAHAP 3] ATRIBUT NON NUMERIK DAN NUMERIK DATASET
    # ============================================================
    list_kolom_non_numerik = df_optimasi.select_dtypes(include=['category', 'object', 'datetime64', 'string']).columns.tolist()
    list_kolom_numerik = df_optimasi.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float32', 'float64']).columns.tolist()

    print(f"\n {C_BOLD}{C_YELLOW}PEMERIKSAAN ATRIBUT DATASET AWAL:{C_RESET}")
    print(line_sg)
    print(f"   • Daftar Kolom Non Numerik : {len(list_kolom_non_numerik)} kolom")
    print(f"   • Daftar Kolom Numerik    : {len(list_kolom_numerik)} atribut")
    print(f"   • Jumlah Atribut Dataset  : {len(list_kolom_non_numerik) + len(list_kolom_numerik)} atribut")

    # ============================================================
    # 3.1 Daftar Atribut Hasil Data Selection
    # ============================================================
    selected_columns = [
        # Lama Menunggak
        'TglPKB', 'TglPKBLalu',
        # Karakteristik transaksi
        "PokokPKB",
        # Komponen denda
        "DendaPKB",
        # Riwayat Pembayaran
        "PokokPKBT1", "PokokPKBT2", "PokokPKBT3", "PokokPKBT4", "PokokPKBT5",
        # Karakteristik kendaraan
        "Jenis_Kendaraan", "Warna_TNKB", "Golongan", "Jenis_Pendaftaran", "TahunPembuatan", "Fungsi",
        # Informasi wilayah
        "Lokasi_Bayar", "NamaWilayah"
    ]

    available_selected_columns = [col for col in selected_columns if col in df_optimasi.columns]
    missing_selected_columns = [col for col in selected_columns if col not in df_optimasi.columns]

    print(f"\n {C_BOLD}{C_YELLOW}3.1 DAFTAR ATRIBUT HASIL DATA SELECTION:{C_RESET}")
    print(line_sg)
    print(f"   • Atribut yang tersedia ({len(available_selected_columns)} atribut):")
    for i, col in enumerate(available_selected_columns, start=1):
        print(f"     {i:>2}. {C_GREEN}{col:<25}{C_RESET} (Type: {df_optimasi[col].dtype})")

    if missing_selected_columns:
        print(f"   • Atribut yang tidak ditemukan: {C_RED}{missing_selected_columns}{C_RESET}")

    # ============================================================
    # 3.2 Membentuk Dataset Hasil Seleksi
    # ============================================================
    df_selected = df_optimasi[available_selected_columns].copy()

    print(f"\n {C_BOLD}{C_YELLOW}3.2 HASIL DATA SELECTION (df_selected):{C_RESET}")
    print(line_sg)
    print(f"   • Jumlah Record  : {C_BOLD}{df_selected.shape[0]:,}{C_RESET}")
    print(f"   • Jumlah Atribut : {C_BOLD}{df_selected.shape[1]}{C_RESET} atribut (Reduksi dari {df_optimasi.shape[1]} kolom)")

    # ============================================================
    # 3.3 Validasi Dataset Hasil Seleksi
    # ============================================================
    missing_val_sum = df_selected.isnull().sum().sum()
    duplicate_val_sum = df_selected.duplicated().sum()

    print(f"\n {C_BOLD}{C_YELLOW}3.3 VALIDASI DATASET HASIL SELEKSI:{C_RESET}")
    print(line_sg)
    print(f"   • Jumlah Missing Value   : {missing_val_sum}")
    print(f"   • Jumlah Kombinasi Sama  : {duplicate_val_sum:,} record")
    print(f"   {C_DIM}* Catatan Penting: Hasil duplicated() jangan langsung digunakan untuk menghapus data.{C_RESET}")
    print(f"   {C_DIM}  Kesamaan kombinasi atribut hasil seleksi tidak otomatis merupakan duplikasi aktual.{C_RESET}")

    # ============================================================
    # 3.4 Menyimpan Dataset Hasil Seleksi
    # ============================================================
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    path_dataset_selection_parquet = os.path.join(output_dir, f"{date_str}_dataset_hasil_selection.parquet")
    path_dataset_selection_csv = os.path.join(output_dir, f"{date_str}_df_selected.csv")
    cache_selected_parquet = os.path.join(cache_dir, "selected_features_dataset.parquet")

    df_selected.to_parquet(path_dataset_selection_parquet, index=False)
    df_selected.to_csv(path_dataset_selection_csv, index=False)
    df_selected.to_parquet(cache_selected_parquet, index=False)

    exec_duration = round(time.time() - start_time, 2)

    print(f"\n {C_BOLD}{C_YELLOW}3.4 MENYIMPAN DATASET HASIL SELEKSI:{C_RESET}")
    print(line_sg)
    print(f"   • Parquet Output : {C_BOLD}{C_GREEN}{path_dataset_selection_parquet}{C_RESET}")
    print(f"   • CSV Output     : {C_BOLD}{C_GREEN}{path_dataset_selection_csv}{C_RESET}")
    print(f"   • Cache Parquet  : {C_BOLD}{C_GREEN}{cache_selected_parquet}{C_RESET}")
    print(f"   • Waktu Eksekusi : {exec_duration} detik")

    print("\n" + line_db)
    print(f" {C_BOLD}{C_GREEN}STATUS TAHAP 3: ✅ DATA SELECTION PERSIS KODINGAN PENELITIAN KANG YAYAT SELESAI 100%!{C_RESET}")
    print(line_db + "\n")

    report = {
        "jumlah_awal_col": df_optimasi.shape[1],
        "jumlah_selected_col": df_selected.shape[1],
        "available_selected_columns": available_selected_columns,
        "missing_val_sum": missing_val_sum,
        "duplicate_val_sum": duplicate_val_sum,
        "path_dataset_selection_parquet": path_dataset_selection_parquet,
        "path_dataset_selection_csv": path_dataset_selection_csv,
        "cache_selected_parquet": cache_selected_parquet,
        "exec_duration": exec_duration
    }

    return df_selected, report


if __name__ == "__main__":
    select_features_for_kmeans()
