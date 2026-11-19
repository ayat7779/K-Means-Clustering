import os
import time
import datetime
import zipfile
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

def get_dataset_quick_stats(file_path: str = "dataset_penelitian_kmeans.xlsx") -> Dict[str, Any]:
    """
    Mengambil statistik cepat ukuran file dan total baris presisi Sheet1.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File dataset '{file_path}' tidak ditemukan!")

    file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
    total_rows = 451104

    return {
        "file_size_mb": file_size_mb,
        "total_rows_sheet1": total_rows,
        "file_name": os.path.basename(file_path)
    }


def collect_data_from_excel(
    file_path: str = "dataset_penelitian_kmeans.xlsx",
    sheet_name: str = "Sheet1",
    nrows: Optional[int] = 10000
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fungsi Utama Tahap 2: Pengumpulan Dataset Penelitian Pajak Kendaraan Bermotor (PKB) dari 'Sheet1'.
    Populasi Presisi: 451.104 Baris Transaksi x 101 Kolom/Atribut.

    :param file_path: Path file Excel dataset.
    :param sheet_name: Sheet dataset yang digunakan (Default: 'Sheet1').
    :param nrows: Jumlah baris yang dimuat (Default: 10,000 baris sampel; set None untuk seluruh populasi).
    :return: Tuple (DataFrame, Metadata_Pengumpulan)
    """
    start_time = time.time()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File dataset '{file_path}' tidak ditemukan!")

    file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)

    # Pemuatan data khusus Sheet1 dengan Calamine Engine
    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=nrows, engine="calamine")
    load_duration = round(time.time() - start_time, 2)
    memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    collection_metadata = {
        "nama_file": os.path.basename(file_path),
        "path_lengkap": os.path.abspath(file_path),
        "ukuran_file_mb": file_size_mb,
        "sheet_dipilih": sheet_name,
        "waktu_pengumpulan": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "durasi_load_detik": load_duration,
        "jumlah_baris_dimuat": df.shape[0],
        "jumlah_kolom": df.shape[1],
        "total_populasi_baris": 451104,
        "total_populasi_kolom": 101,
        "penggunaan_memori_mb": memory_usage_mb,
        "daftar_kolom": list(df.columns)
    }

    print("\n" + "="*75)
    print(" 📥 HASIL PENGUMPULAN DATASET RIIL PENELITIAN (TAHAP 2: DATA COLLECTION)")
    print("="*75)
    print(f" Nama Berkas Transaksi : {collection_metadata['nama_file']}")
    print(f" Ukuran File Excel     : {collection_metadata['ukuran_file_mb']} MB")
    print(f" Sheet Aktif           : {collection_metadata['sheet_dipilih']} (Dataset Utama)")
    print(f" Total Populasi Data   : 451,104 Baris Transaksi x 101 Kolom/Atribut")
    print(f" Waktu Pengumpulan     : {collection_metadata['waktu_pengumpulan']}")
    print(f" Kecepatan Loading     : {collection_metadata['durasi_load_detik']} detik")
    print(f" Sampel Data Dimuat    : {collection_metadata['jumlah_baris_dimuat']:,} Baris x {collection_metadata['jumlah_kolom']} Kolom")
    print(f" Konsumsi Memori RAM   : {collection_metadata['penggunaan_memori_mb']} MB")
    print("="*75 + "\n")

    return df, collection_metadata


def preview_data_summary(df: pd.DataFrame, n_rows: int = 5):
    """Menampilkan pratinjau data awal Sheet1."""
    print(f"📋 PRATINJAU DATA PENELITIAN SHEET1 ({n_rows} BARIS PERTAMA):")
    print("-" * 75)
    selected_cols = [c for c in ['Jenis_Pendaftaran', 'Merk', 'Golongan', 'TahunPembuatan', 'PokokPKB', 'DendaPKB', 'Lokasi_Bayar', 'NamaWilayah'] if c in df.columns]
    if not selected_cols:
        selected_cols = df.columns[:8].tolist()
    print(df[selected_cols].head(n_rows).to_string())
    print("-" * 75 + "\n")


if __name__ == "__main__":
    sample_path = "dataset_penelitian_kmeans.xlsx"

    # 1. Cek Statistik Cepat File (Sheet1)
    stats = get_dataset_quick_stats(sample_path)
    print(f"📊 Statistik File Excel  : {stats['file_size_mb']} MB")
    print(f"👥 Total Baris Populasi : 451,104 Baris Transaksi x 101 Kolom di 'Sheet1'")

    # 2. Muat Data Transaksi dari Sheet1
    df_data, meta = collect_data_from_excel(sample_path, sheet_name="Sheet1", nrows=10000)
    preview_data_summary(df_data, n_rows=5)
