import os
import sys
import time
import datetime
import zipfile
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


def run_full_data_collection_analysis(
    file_path: str = "dataset_penelitian_kmeans.xlsx",
    sheet_name: str = "Sheet1",
    nrows: Optional[int] = 20000
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    [TAHAP 2] PENGUMPULAN DATA - METODOLOGI PENELITIAN KANG YAYAT:
    Populasi Presisi: 451.104 Baris Transaksi x 101 Kolom/Atribut
    """
    start_time = time.time()
    W = 85
    line_db = C_CYAN + "═" * W + C_RESET
    line_sg = C_DIM + "─" * W + C_RESET

    print("\n" + line_db)
    print(f" {C_BOLD}{C_MAGENTA}📥 [TAHAP 2] PENGUMPULAN DATA & PROFILING (METODOLOGI KANG YAYAT){C_RESET}")
    print(f" {C_DIM}Tim Pengembang: Mang Ewok (Senior DS) & Kang Yayat | Dataset: Transaksi PKB{C_RESET}")
    print(line_db)

    # ============================================================
    # 2.1 & 2.2 DEFINISI LOKASI DATASET
    # ============================================================
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File dataset '{file_path}' tidak ditemukan!")

    abs_path = os.path.abspath(file_path)
    file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
    output_dir = os.path.join(os.getcwd(), "Hasil_Penelitian_PKB", "01_Dataset")
    cache_dir = os.path.join(os.getcwd(), ".cache_dataset")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n {C_BOLD}{C_YELLOW}2.1 & 2.2 DEFINISI LOKASI DATASET:{C_RESET}")
    print(line_sg)
    print(f"   • Nama Berkas Dataset : {C_BOLD}{C_CYAN}{os.path.basename(file_path)}{C_RESET}")
    print(f"   • Path Absolut        : {C_DIM}{abs_path}{C_RESET}")
    print(f"   • Ukuran File         : {C_BOLD}{C_GREEN}{file_size_mb} MB{C_RESET}")

    # ============================================================
    # 2.3 MEMBACA DATASET EXCEL
    # ============================================================
    print(f"\n {C_BOLD}{C_YELLOW}2.3 MEMBACA DATASET EXCEL:{C_RESET}")
    print(line_sg)
    t0 = time.time()
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, nrows=nrows, engine="calamine")
    t1 = time.time()
    load_duration = round(t1 - t0, 2)
    total_populasi = 451104
    total_kolom_populasi = 101

    print(f"   • Engine Pembacaan    : Calamine Engine (High Performance Rust Parser)")
    print(f"   • Kecepatan Reading   : {load_duration} detik")
    print(f"   • Total Baris Populasi: {C_BOLD}{total_populasi:,} Baris Transaksi{C_RESET}")
    print(f"   • Total Atribut       : {C_BOLD}{total_kolom_populasi} Kolom/Atribut{C_RESET}")
    print(f"   • Sampel Dimuat       : {C_BOLD}{len(df_raw):,} Baris x {df_raw.shape[1]} Kolom{C_RESET}")

    # ============================================================
    # 2.4 MENYIMPAN DATASET AWAL (PARQUET CACHE)
    # ============================================================
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    path_dataset_awal_parquet = os.path.join(output_dir, f"{date_str}_dataset_awal.parquet")
    cache_parquet_awal = os.path.join(cache_dir, "raw_dataset_cache.parquet")

    df_raw.to_parquet(path_dataset_awal_parquet, index=False)
    df_raw.to_parquet(cache_parquet_awal, index=False)

    memory_awal = df_raw.memory_usage(deep=True).sum() / (1024**2)

    print(f"\n {C_BOLD}{C_YELLOW}2.4 MENYIMPAN DATASET AWAL:{C_RESET}")
    print(line_sg)
    print(f"   • Dataset awal disimpan : {C_GREEN}{path_dataset_awal_parquet}{C_RESET}")
    print(f"   • Penggunaan Memori Awal: {C_BOLD}{memory_awal:.2f} MB{C_RESET}")

    # ============================================================
    # 2.5 OPTIMASI TIPE DATA
    # ============================================================
    df_optimasi = df_raw.copy()

    for col in df_optimasi.columns:
        col_type = df_optimasi[col].dtype
        # Optimasi integer
        if pd.api.types.is_integer_dtype(col_type):
            c_min = df_optimasi[col].min()
            c_max = df_optimasi[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df_optimasi[col] = df_optimasi[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df_optimasi[col] = df_optimasi[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df_optimasi[col] = df_optimasi[col].astype(np.int32)
        # Optimasi float
        elif pd.api.types.is_float_dtype(col_type):
            df_optimasi[col] = df_optimasi[col].astype(np.float32)
        # Optimasi object menjadi category
        elif pd.api.types.is_object_dtype(col_type):
            nunique = df_optimasi[col].nunique(dropna=False)
            total = len(df_optimasi[col])
            if total > 0 and (nunique / total) < 0.5:
                df_optimasi[col] = df_optimasi[col].astype("category")

    memory_optimasi = df_optimasi.memory_usage(deep=True).sum() / (1024**2)
    pct_optimasi = (1 - (memory_optimasi / memory_awal)) * 100

    print(f"\n {C_BOLD}{C_YELLOW}2.5 HASIL OPTIMASI TIPE DATA:{C_RESET}")
    print(line_sg)
    print(f"   • Memori Sebelum Optimasi : {memory_awal:.2f} MB")
    print(f"   • Memori Setelah Optimasi: {C_BOLD}{C_GREEN}{memory_optimasi:.2f} MB{C_RESET}")
    print(f"   • Persentase Optimasi     : {C_BOLD}{C_GREEN}{pct_optimasi:.2f} %{C_RESET} penghematan RAM")

    # ============================================================
    # 2.6 PROFIL DATASET AWAL
    # ============================================================
    print(f"\n {C_BOLD}{C_YELLOW}2.6 PROFIL DATASET AWAL:{C_RESET}")
    print(line_sg)
    print(f"   • Total Baris Populasi : {C_BOLD}{total_populasi:,} Baris Transaksi{C_RESET}")
    print(f"   • Total Atribut/Kolom  : {C_BOLD}{total_kolom_populasi} Kolom/Atribut{C_RESET}")

    # ============================================================
    # 2.7 DAFTAR ATRIBUT NUMERIK DAN NON NUMERIK (TABEL 5 KOLOM)
    # ============================================================
    list_kolom_non_numerik = df_optimasi.select_dtypes(include=['category', 'object', 'datetime64', 'string']).columns.tolist()
    list_kolom_numerik = df_optimasi.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float32', 'float64']).columns.tolist()

    df_tipe_tabel = pd.DataFrame([
        {
            "No": 1,
            "Tipe Data": "Numerik",
            "Jumlah": len(list_kolom_numerik),
            "Data": "Finansial, Sanksi Denda, Tunggakan T1-T5, Spesifikasi, Durasi",
            "Nama Data": ", ".join(list_kolom_numerik[:15]) + f"... (+{len(list_kolom_numerik)-15} kolom)"
        },
        {
            "No": 2,
            "Tipe Data": "Non Numerik",
            "Jumlah": len(list_kolom_non_numerik),
            "Data": "Kategorikal Transaksi, Identitas PII, Wilayah, Timestamp",
            "Nama Data": ", ".join(list_kolom_non_numerik[:15]) + f"... (+{len(list_kolom_non_numerik)-15} kolom)"
        }
    ])

    print(f"\n {C_BOLD}{C_YELLOW}2.7 DAFTAR ATRIBUT NUMERIK DAN NON NUMERIK (TABEL 5 KOLOM):{C_RESET}")
    print(line_sg)
    print(df_tipe_tabel.to_string(index=False))

    # ============================================================
    # 2.12 REKAPITULASI MISSING VALUES
    # ============================================================
    missing_count = df_optimasi.isnull().sum()
    missing_filtered = missing_count[missing_count > 0]

    df_missing_summary = pd.DataFrame({
        'Nama Kolom': missing_filtered.index,
        'Tipe Data': df_optimasi[missing_filtered.index].dtypes.values,
        'Jumlah Missing': missing_filtered.values,
        'Persentase (%)': (missing_filtered.values / len(df_optimasi)) * 100
    }).sort_values(by='Jumlah Missing', ascending=False).reset_index(drop=True)

    print(f"\n {C_BOLD}{C_YELLOW}2.12 REKAPITULASI MISSING VALUES DATASET AWAL:{C_RESET}")
    print(line_sg)
    print(f"   • Total Kolom Mengandung Missing Values: {len(df_missing_summary)} dari {df_optimasi.shape[1]} kolom")

    # ============================================================
    # MENYIMPAN HASIL DATASET OPTIMASI (CSV)
    # ============================================================
    path_dataset_optimasi_csv = os.path.join(output_dir, f"{date_str}_df_optimasi.csv")
    df_optimasi.to_csv(path_dataset_optimasi_csv, index=False)

    exec_duration = round(time.time() - start_time, 2)

    print("\n" + line_db)
    print(f" {C_BOLD}{C_GREEN}STATUS TAHAP 2: ✅ PENGUMPULAN DATA SELESAI 100%! SIAP KA TAHAP 3 (SELEKSI DATA){C_RESET}")
    print(line_db + "\n")

    report = {
        "file_size_mb": file_size_mb,
        "total_populasi_baris": total_populasi,
        "total_populasi_kolom": total_kolom_populasi,
        "raw_shape": df_raw.shape,
        "optimasi_shape": df_optimasi.shape,
        "memory_awal_mb": memory_awal,
        "memory_optimasi_mb": memory_optimasi,
        "pct_optimasi": pct_optimasi,
        "list_kolom_non_numerik": list_kolom_non_numerik,
        "list_kolom_numerik": list_kolom_numerik,
        "df_missing_summary": df_missing_summary,
        "path_dataset_awal_parquet": path_dataset_awal_parquet,
        "path_dataset_optimasi_csv": path_dataset_optimasi_csv,
        "exec_duration": exec_duration
    }

    return df_raw, df_optimasi, report


if __name__ == "__main__":
    run_full_data_collection_analysis()
