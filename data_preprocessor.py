import os
import sys
import time
import json
import datetime
import pandas as pd
import numpy as np
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


def preprocess_dataset(
    df_input: Optional[pd.DataFrame] = None,
    file_path: str = "dataset_penelitian_kmeans.xlsx",
    handle_missing_strategy: str = "dropna", # 'dropna' atau 'impute' saluyu metodologi Kang Yayat
    validate_dates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    [TAHAP 4] DATA PREPROCESSING - PERSIS SALUYU KODINGAN PENELITIAN KANG YAYAT:
    4.1 Membaca Dataset Hasil Seleksi
    4.2 Data Cleaning (String Strip tanpa Regex Buta)
    4.3 Pemeriksaan Missing Value
    4.4 Penanganan Missing Value (Metodologis dropna/impute)
    4.5 Validasi Duplikasi (Audit tanpa asal drop_duplicates)
    4.6 Deteksi Outlier Menggunakan IQR (Q1, Q3, IQR, Lower & Upper Bounds)
    4.7 Validasi dan Koreksi TglPKBLalu dan TglPKB (Konsistensi Logika Tanggal)
    4.8 Menyimpan Hasil Preprocessing Data
    """
    start_time = time.time()
    W = 85
    line_db = C_CYAN + "═" * W + C_RESET
    line_sg = C_DIM + "─" * W + C_RESET

    print("\n" + line_db)
    print(f" {C_BOLD}{C_MAGENTA}🧹 [TAHAP 4] DATA PREPROCESSING (METODOLOGI PENELITIAN KANG YAYAT){C_RESET}")
    print(f" {C_DIM}Tim Pengembang: Mang Ewok (Senior DS) & Kang Yayat | Algoritma K-Means{C_RESET}")
    print(line_db)

    # ============================================================
    # 4.1 Membaca Dataset Hasil Seleksi
    # ============================================================
    if df_input is None:
        cache_path = os.path.join(os.getcwd(), ".cache_dataset", "selected_features_dataset.parquet")
        if os.path.exists(cache_path):
            print(f" 📦 Memuat dataset hasil seleksi dari Parquet Cache: {C_DIM}{os.path.basename(cache_path)}{C_RESET}")
            df_preprocessed = pd.read_parquet(cache_path)
        else:
            print(f" 📖 Memuat data dari Excel: {C_DIM}{os.path.basename(file_path)}{C_RESET}")
            df_preprocessed = pd.read_excel(file_path, sheet_name="Sheet1", nrows=20000, engine="calamine")
    else:
        df_preprocessed = df_input.copy()

    jumlah_awal = len(df_preprocessed)
    print(f" Dataset berhasil dimuat:\n - {jumlah_awal:,} record\n - {df_preprocessed.shape[1]} atribut.")

    # ============================================================
    # 4.2 Data Cleaning (Pembersihan Spasi Kolom Teks)
    # ============================================================
    text_columns = df_preprocessed.select_dtypes(include=["object", "category", "string"]).columns
    for col in text_columns:
        df_preprocessed[col] = df_preprocessed[col].astype("string").str.strip()

    print(f" {C_GREEN}✅ Data cleaning dasar selesai (String strip spasi tanpa regex membabi buta).{C_RESET}")

    # Drop kolom dengan missing value > 50% (seperti TglFiskal, TglSKET) agar dropna() tidak membuang 100% baris data
    missing_pct_col = (df_preprocessed.isnull().mean()) * 100
    high_missing_cols = missing_pct_col[missing_pct_col > 50.0].index.tolist()
    if high_missing_cols:
        df_preprocessed = df_preprocessed.drop(columns=high_missing_cols)

    # ============================================================
    # 4.3 Pemeriksaan Missing Value
    # ============================================================
    missing_summary = pd.DataFrame({
        "Jumlah Missing": df_preprocessed.isnull().sum(),
        "Persentase (%)": (df_preprocessed.isnull().mean() * 100)
    })
    missing_summary_filtered = missing_summary[missing_summary["Jumlah Missing"] > 0].sort_values("Jumlah Missing", ascending=False)

    # ============================================================
    # 4.4 Penanganan Missing Value
    # ============================================================
    if handle_missing_strategy == "dropna":
        df_preprocessed_final = df_preprocessed.dropna().copy()
    else:
        df_preprocessed_final = df_preprocessed.copy()
        for col in df_preprocessed_final.columns:
            if df_preprocessed_final[col].isnull().sum() > 0:
                if np.issubdtype(df_preprocessed_final[col].dtype, np.number):
                    df_preprocessed_final[col] = df_preprocessed_final[col].fillna(df_preprocessed_final[col].median())
                else:
                    df_preprocessed_final[col] = df_preprocessed_final[col].fillna(df_preprocessed_final[col].mode()[0] if not df_preprocessed_final[col].empty else "Unknown")

    jumlah_akhir = len(df_preprocessed_final)
    jumlah_dihapus = jumlah_awal - jumlah_akhir
    pct_dihapus = (jumlah_dihapus / jumlah_awal * 100) if jumlah_awal > 0 else 0.0

    print(f"\n {C_BOLD}{C_YELLOW}4.4 HASIL PENANGANAN MISSING VALUE:{C_RESET}")
    print(line_sg)
    print(f"   • Data sebelum : {jumlah_awal:,}")
    print(f"   • Data sesudah : {jumlah_akhir:,}")
    print(f"   • Data hilang  : {jumlah_dihapus:,}")
    print(f"   • Persentase data hilang: {C_BOLD}{pct_dihapus:.4f}%{C_RESET}")

    # ============================================================
    # 4.5 Validasi Duplikasi
    # ============================================================
    duplicate_count = df_preprocessed_final.duplicated().sum()
    print(f"\n {C_BOLD}{C_YELLOW}4.5 VALIDASI DUPLIKASI:{C_RESET}")
    print(line_sg)
    print(f"   • Jumlah baris dengan kombinasi atribut sama: {C_BOLD}{duplicate_count:,}{C_RESET} record.")
    print(f"   {C_DIM}* Catatan: Data tidak langsung dihapus via drop_duplicates() karena kesamaan atribut tidak otomatis menunjukkan duplikasi transaksi aktual.{C_RESET}")

    # ============================================================
    # 4.6 Deteksi Outlier Menggunakan IQR
    # ============================================================
    numerical_cols = df_preprocessed_final.select_dtypes(include=np.number).columns.tolist()
    outlier_summary = {}
    outlier_mask_all = pd.Series(False, index=df_preprocessed_final.index)

    for col in numerical_cols:
        Q1 = df_preprocessed_final[col].quantile(0.25)
        Q3 = df_preprocessed_final[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        mask = (df_preprocessed_final[col] < lower_bound) | (df_preprocessed_final[col] > upper_bound)

        outlier_summary[col] = {
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Jumlah Outlier": int(mask.sum())
        }
        outlier_mask_all |= mask

    outlier_summary_df = pd.DataFrame(outlier_summary).T.sort_values("Jumlah Outlier", ascending=False)
    total_records_with_outlier = int(outlier_mask_all.sum())

    print(f"\n {C_BOLD}{C_YELLOW}4.6 DETEKSI OUTLIER MENGGUNAKAN IQR:{C_RESET}")
    print(line_sg)
    print(f"   • Jumlah data unik yang memiliki minimal satu outlier: {C_BOLD}{total_records_with_outlier:,}{C_RESET}")
    print("   • Top 5 Atribut dengan Outlier Terbanyak:")
    for idx, row_iqr in outlier_summary_df.head(5).iterrows():
        print(f"     - {idx:<22}: {int(row_iqr['Jumlah Outlier']):>6,} outliers | IQR Bounds: [{row_iqr['Lower Bound']:.1f} s/d {row_iqr['Upper Bound']:.1f}]")

    # ============================================================
    # 4.7 Validasi dan Koreksi TglPKBLalu dan TglPKB
    # ============================================================
    initial_invalid_dates = 0
    if validate_dates and 'TglPKB' in df_preprocessed_final.columns and 'TglPKBLalu' in df_preprocessed_final.columns:
        print(f"\n {C_BOLD}{C_YELLOW}4.7 VALIDASI DAN KOREKSI TGLPKBLALU DAN TGLPKB:{C_RESET}")
        print(line_sg)
        df_preprocessed_final['TglPKB'] = pd.to_datetime(df_preprocessed_final['TglPKB'], errors='coerce')
        df_preprocessed_final['TglPKBLalu'] = pd.to_datetime(df_preprocessed_final['TglPKBLalu'], errors='coerce')

        rows_before_date_val = df_preprocessed_final.shape[0]
        invalid_dates_mask = df_preprocessed_final['TglPKBLalu'] > df_preprocessed_final['TglPKB']
        initial_invalid_dates = int(invalid_dates_mask.fillna(False).sum())

        print(f"   • Jumlah data sebelum validasi tanggal: {rows_before_date_val:,}")
        if initial_invalid_dates > 0:
            df_preprocessed_final = df_preprocessed_final[~invalid_dates_mask.fillna(False)].copy()
            print(f"   • {C_RED}Ditemukan dan dihapus {initial_invalid_dates:,} entri dimana TglPKBLalu lebih baru dari TglPKB.{C_RESET}")
        else:
            print(f"   • {C_GREEN}Tidak ada entri dimana TglPKBLalu lebih baru dari TglPKB. Data tanggal konsisten.{C_RESET}")
        print(f"   • Jumlah data setelah validasi tanggal: {df_preprocessed_final.shape[0]:,}")

    # ============================================================
    # 4.8 Menyimpan Hasil Preprocessing Data
    # ============================================================
    cache_dir = os.path.join(os.getcwd(), ".cache_dataset")
    os.makedirs(cache_dir, exist_ok=True)
    preprocessed_cache_parquet = os.path.join(cache_dir, "preprocessed_dataset.parquet")
    preprocessed_cache_csv = os.path.join(cache_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d')}_df_preprocessed_final.csv")

    df_preprocessed_final.to_parquet(preprocessed_cache_parquet, index=False)
    df_preprocessed_final.to_csv(preprocessed_cache_csv, index=False)

    exec_duration = round(time.time() - start_time, 2)

    print(f"\n {C_BOLD}{C_YELLOW}4.8 MENYIMPAN HASIL PREPROCESSING DATA:{C_RESET}")
    print(line_sg)
    print(f"   • Berkas CSV Simpan    : {C_BOLD}{C_GREEN}{os.path.basename(preprocessed_cache_csv)}{C_RESET}")
    print(f"   • Berkas Parquet Cache : {C_BOLD}{C_GREEN}{os.path.basename(preprocessed_cache_parquet)}{C_RESET}")
    print(f"   • Dimensi Akhir Data   : {df_preprocessed_final.shape[0]:,} record x {df_preprocessed_final.shape[1]} atribut.")
    print(f"   • Waktu Pemprosesan    : {exec_duration} detik")

    print("\n" + line_db)
    print(f" {C_BOLD}{C_GREEN}STATUS TAHAP 4: ✅ DATA PREPROCESSING SAMA PERSIS KODINGAN PENELITIAN KANG YAYAT SELESAI 100%!{C_RESET}")
    print(line_db + "\n")

    report = {
        "jumlah_awal": jumlah_awal,
        "jumlah_akhir": len(df_preprocessed_final),
        "jumlah_dihapus": jumlah_dihapus,
        "persentase_dihapus": pct_dihapus,
        "duplicate_count": duplicate_count,
        "outlier_summary_df": outlier_summary_df,
        "total_records_with_outlier": total_records_with_outlier,
        "initial_invalid_dates": initial_invalid_dates,
        "output_path_csv": preprocessed_cache_csv,
        "output_path_parquet": preprocessed_cache_parquet,
        "missing_summary_filtered": missing_summary_filtered
    }

    return df_preprocessed_final, report


if __name__ == "__main__":
    preprocess_dataset()
