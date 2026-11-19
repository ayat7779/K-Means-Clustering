import os
import sys
import json
import time
import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Terminal Color Support
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


def get_available_numeric_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan matriks evaluasi kelayakan seluruh atribut numerik dalam dataset.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    stats_list = []
    for i, col in enumerate(num_cols):
        series = df[col].dropna()
        std_val = series.std() if len(series) > 1 else 0
        mean_val = series.mean() if len(series) > 0 else 0
        skew_val = series.skew() if len(series) > 2 else 0
        missing_cnt = df[col].isnull().sum()
        missing_pct = (missing_cnt / len(df)) * 100
        n_unique = series.nunique()

        # Rekomendasi Kelayakan untuk K-Means
        if n_unique <= 1 or std_val == 0:
            recom = "❌ Abaikan (Konstanta)"
            score = 0
        elif missing_pct > 50:
            recom = "❌ Abaikan (High Missing)"
            score = 1
        elif col in ['PokokPKB', 'DendaPKB']:
            recom = "⭐ SANGAT REKOMENDASI (Inti Finansial)"
            score = 5
        elif col in ['TahunPembuatan', 'NilaiJual', 'IsiCylinder']:
            recom = "✅ REKOMENDASI (Spesifikasi Kendaraan)"
            score = 4
        elif col in ['PokokPKBT1', 'PokokPKBT2', 'PokokPKBT3', 'PokokPKBT4', 'PokokPKBT5']:
            recom = "🔹 OPSIONAL (Tunggakan PKB)"
            score = 3
        elif col in ['Hari', 'Bulan', 'Tahun']:
            recom = "🔹 OPSIONAL (Durasi Waktu)"
            score = 3
        else:
            recom = "🔹 OPSIONAL"
            score = 2

        stats_list.append({
            "No": i + 1,
            "Atribut": col,
            "Tipe": str(df[col].dtype),
            "Mean": round(mean_val, 2),
            "Std_Dev": round(std_val, 2),
            "Missing_%": round(missing_pct, 2),
            "Unique": n_unique,
            "Skewness": round(skew_val, 2),
            "Rekomendasi": recom,
            "_score": score
        })

    df_result = pd.DataFrame(stats_list)
    return df_result


def display_numeric_feature_selection_menu(df_eval: pd.DataFrame):
    """
    Menampilkan menu interaktif daftar atribut numerik yang tersedia.
    """
    W = 85
    line_db = C_CYAN + "═" * W + C_RESET
    line_sg = C_DIM + "─" * W + C_RESET

    print("\n" + line_db)
    print(f" {C_BOLD}{C_MAGENTA}🎛️  FITUR PENELITI: SELEKSI ATRIBUT NUMERIK PIKEUN K-MEANS CLUSTERING{C_RESET}")
    print(f" {C_DIM}Pengembang: Mang Ewok & Kang Yayat | Modul Interaktif Seleksi Fitur{C_RESET}")
    print(line_db)

    print(f"\n {C_BOLD}{C_YELLOW}📋 DAFTAR ATRIBUT NUMERIK YANG TERSEDIA:{C_RESET}\n")
    print(f" {'No':<4} | {'Nama Atribut':<22} | {'Missing':<8} | {'Unique':<8} | {'Status Rekomendasi'}")
    print(line_sg)

    for idx, row in df_eval.iterrows():
        no_str = f"{row['No']:>2}."
        col_str = f"{row['Atribut']:<22}"
        miss_str = f"{row['Missing_%']:>5.1f}%"
        uniq_str = f"{row['Unique']:>6,}"
        recom_str = row['Rekomendasi']

        if "SANGAT REKOMENDASI" in recom_str:
            recom_fmt = f"{C_GREEN}{recom_str}{C_RESET}"
        elif "REKOMENDASI" in recom_str:
            recom_fmt = f"{C_CYAN}{recom_str}{C_RESET}"
        elif "Abaikan" in recom_str:
            recom_fmt = f"{C_DIM}{recom_str}{C_RESET}"
        else:
            recom_fmt = f"{C_YELLOW}{recom_str}{C_RESET}"

        print(f" {no_str} | {C_BOLD}{col_str}{C_RESET} | {miss_str:<8} | {uniq_str:<8} | {recom_fmt}")
    print(line_sg)


def select_numeric_features(
    df: pd.DataFrame,
    selected_indices: Optional[List[int]] = None,
    save_config: bool = True
) -> Tuple[List[str], pd.DataFrame]:
    """
    Fungsi Seleksi Atribut Numerik Pilihan Kang Yayat.

    :param df: DataFrame input.
    :param selected_indices: List nomor indeks atribut yang dipilih (Misal: [1, 2, 3, 4, 5]).
           Jika None, akan memilih otomatis fitur yang direkomendasikan.
    :param save_config: Menyimpan preferensi ke file JSON config cache.
    :return: Tuple (List_Nama_Fitur_Terpilih, DataFrame_Sub-Set_Numerik)
    """
    df_eval = get_available_numeric_attributes(df)
    display_numeric_feature_selection_menu(df_eval)

    available_cols = df_eval['Atribut'].tolist()

    # Mode Pilihan Otomatis (Default Rekomendasi Peneliti)
    if selected_indices is None:
        # Pilih fitur dengan skor rekomendasi >= 3
        recommended_rows = df_eval[df_eval['_score'] >= 3]
        chosen_cols = recommended_rows['Atribut'].tolist()
        chosen_indices = recommended_rows['No'].tolist()
    else:
        chosen_indices = selected_indices
        chosen_cols = [available_cols[i - 1] for i in selected_indices if 1 <= i <= len(available_cols)]

    # Validasi minimal 2 fitur numerik untuk K-Means
    if len(chosen_cols) < 2:
        print(f"\n{C_RED}[WARNING] Pilihan fitur terlalu sedikit. Menambahkan PokokPKB & DendaPKB secara otomatis.{C_RESET}")
        for fallback in ['PokokPKB', 'DendaPKB']:
            if fallback not in chosen_cols and fallback in available_cols:
                chosen_cols.append(fallback)

    df_selected_numeric = df[chosen_cols].copy()

    # Tampilkan Ringkasan Fitur Terpilih
    print(f"\n {C_BOLD}{C_GREEN}✅ FITUR NUMERIK TERPILIH PIKEUN KLUSTERISASI K-MEANS ({len(chosen_cols)} ATRIBUT):{C_RESET}")
    print("-" * 85)
    for idx, col_name in enumerate(chosen_cols, 1):
        print(f"   {idx}. {C_BOLD}{C_CYAN}{col_name:<22}{C_RESET} (Type: {df[col_name].dtype})")
    print("-" * 85)

    # Simpan Konfigurasi Fitur ke File JSON
    if save_config:
        cache_dir = os.path.join(os.getcwd(), ".cache_dataset")
        os.makedirs(cache_dir, exist_ok=True)
        config_path = os.path.join(cache_dir, "selected_numeric_config.json")

        config_data = {
            "selected_numeric_features": chosen_cols,
            "selected_indices": chosen_indices,
            "total_selected": len(chosen_cols),
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f" 💾 Konfigurasi fitur tersimpan di: {C_DIM}{config_path}{C_RESET}\n")

    return chosen_cols, df_selected_numeric


if __name__ == "__main__":
    cache_path = os.path.join(os.getcwd(), ".cache_dataset", "selected_features_dataset.parquet")
    if os.path.exists(cache_path):
        df_sample = pd.read_parquet(cache_path)
    else:
        df_sample = pd.read_excel("dataset_penelitian_kmeans.xlsx", sheet_name="Sheet1", nrows=5000, engine="calamine")

    # Pengujian seleksi fitur numerik (Contoh memilih fitur 1, 2, 3, 4, 5)
    selected_cols, df_num = select_numeric_features(df_sample, selected_indices=[1, 2, 3, 4, 5])
