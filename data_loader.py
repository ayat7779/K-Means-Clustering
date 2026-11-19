import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any

def load_excel_data(file_path: str, sheet_name: Any = 0) -> pd.DataFrame:
    """
    Memuat data dari file Excel (.xlsx / .xls) ke dalam Pandas DataFrame.

    :param file_path: Path lokasi file Excel.
    :param sheet_name: Nama sheet atau indeks sheet (default: 0).
    :return: DataFrame berisi data dari file Excel.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"[SUCCESS] Berhasil memuat data dari '{file_path}' (Sheet: {sheet_name}).")
        print(f"          Jumlah Baris: {df.shape[0]}, Jumlah Kolom: {df.shape[1]}")
        return df
    except Exception as e:
        print(f"[ERROR] Gagal memuat file Excel: {e}")
        raise e

def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Melakukan pemeriksaan struktur dan kualitas data awal.

    :param df: Pandas DataFrame.
    :return: Dictionary ringkasan struktur data.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }

    print("\n" + "="*50)
    print("      RINGKASAN EKSPLORASI DATA AWAL (EDA)")
    print("="*50)
    print(f"Dimensi Data        : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"Jumlah Baris Duplikat: {summary['duplicate_rows']}")
    print("\nDetail Kolom & Tipe Data:")

    info_df = pd.DataFrame({
        "Tipe Data": df.dtypes,
        "Jumlah Non-Null": df.notnull().sum(),
        "Jumlah Missing": df.isnull().sum(),
        "% Missing": (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(info_df.to_string())

    return summary

def get_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan statistik deskriptif khusus kolom numerik.

    :param df: Pandas DataFrame.
    :return: DataFrame statistik deskriptif.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("[WARNING] Tidak ditemukan kolom numerik dalam dataset.")
        return pd.DataFrame()

    summary_stats = numeric_df.describe().T
    summary_stats['skewness'] = numeric_df.skew()
    summary_stats['kurtosis'] = numeric_df.kurt()

    print("\n" + "="*50)
    print("      STATISTIK DESKRIPTIF (KOLOM NUMERIK)")
    print("="*50)
    print(summary_stats.round(4).to_string())

    return summary_stats

if __name__ == "__main__":
    # Contoh pembuatan dummy data Excel untuk pengujian
    dummy_data = pd.DataFrame({
        "ID_Responden": [f"R_{i+1:03d}" for i in range(10)],
        "Usia": [25, 30, 45, 35, 22, 50, 40, 28, 33, 48],
        "Pendapatan_Juta": [5.5, 7.0, 15.0, 9.5, 4.0, 18.5, 12.0, 6.5, 8.0, 16.0],
        "Skor_Pengeluaran": [60, 75, 20, 80, 55, 15, 30, 70, 85, 25],
        "Kategori_Member": ["Basic", "Gold", "Platinum", "Gold", "Basic", "Platinum", "Gold", "Basic", "Gold", "Platinum"]
    })

    dummy_file = "sample_data_test.xlsx"
    dummy_data.to_excel(dummy_file, index=False)
    print(f"Data dummy berhasil dibuat: '{dummy_file}'\n")

    # Pengujian fungsi
    df_loaded = load_excel_data(dummy_file)
    inspect_data(df_loaded)
    get_numeric_summary(df_loaded)
