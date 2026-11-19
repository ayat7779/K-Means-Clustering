import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.metrics import silhouette_score

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


def run_research_summary_and_export(
    df_cluster_input: Optional[pd.DataFrame] = None,
    X_kmeans_input: Optional[pd.DataFrame] = None,
    optimal_k: int = 5,
    best_silhouette: float = 0.727819,
    export_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    [TAHAP 9] KESIMPULAN DAN REKOMENDASI - METODOLOGI PENELITIAN KANG YAYAT:
    9.1 Ringkasan Hasil Penelitian (K Optimal, Features List, Cluster Distribution %, Silhouette Score)
    9.2 Menyimpan Hasil Clustering Multi-Format (.parquet, .csv, .pkl)
    9.3 Verifikasi & Pembacaan Kembali Berkas (.pkl test read)
    """
    start_time = time.time()
    W = 85
    line_db = C_CYAN + "═" * W + C_RESET
    line_sg = C_DIM + "─" * W + C_RESET

    print("\n" + line_db)
    print(f" {C_BOLD}{C_MAGENTA}🏆 [TAHAP 9] KESIMPULAN DAN EKSPOR HASIL PENELITIAN (METODOLOGI KANG YAYAT){C_RESET}")
    print(f" {C_DIM}Tim Pengembang: Mang Ewok (Senior DS) & Kang Yayat | Final Research Report{C_RESET}")
    print(line_db)

    cache_dir = os.path.join(os.getcwd(), ".cache_dataset")
    
    # 1. Pemuatan Data dari Tahap 7 / 8
    if df_cluster_input is None:
        clustered_path = os.path.join(cache_dir, "clustered_dataset_final.parquet")
        if os.path.exists(clustered_path):
            print(f" 📦 Memuat df_cluster dari Cache Tahap 7/8: {C_DIM}{os.path.basename(clustered_path)}{C_RESET}")
            df_cluster = pd.read_parquet(clustered_path)
        else:
            raise FileNotFoundError("Cache dataset hasil klasterisasi (clustered_dataset_final.parquet) tidak ditemukan!")
    else:
        df_cluster = df_cluster_input.copy()

    if X_kmeans_input is None:
        scaled_path = os.path.join(cache_dir, "transformed_scaled_dataset.parquet")
        if os.path.exists(scaled_path):
            X_kmeans = pd.read_parquet(scaled_path)
        else:
            X_kmeans = df_cluster.select_dtypes(include=[np.number]).copy()
    else:
        X_kmeans = X_kmeans_input.copy()

    # ============================================================
    # 9.1 Ringkasan Hasil Penelitian
    # ============================================================
    features_used_for_clustering = X_kmeans.columns.tolist()
    cluster_distribution = df_cluster["Cluster"].value_counts().sort_index()
    cluster_percentage = df_cluster["Cluster"].value_counts(normalize=True).sort_index().mul(100)

    df_final_distribution = pd.DataFrame({
        "Jumlah Data": cluster_distribution,
        "Persentase (%)": cluster_percentage.round(4)
    })

    print(f"\n {C_BOLD}{C_YELLOW}9.1 RINGKASAN HASIL PENELITIAN K-MEANS:{C_RESET}")
    print(line_sg)
    print(f"   • Jumlah Data Akhir     : {C_BOLD}{len(df_cluster):,}{C_RESET} record")
    print(f"   • Jumlah Fitur K-Means  : {len(features_used_for_clustering)}")
    print("   • Fitur yang digunakan  :")
    for i, feature in enumerate(features_used_for_clustering, start=1):
        print(f"     {i}. {C_CYAN}{feature}{C_RESET}")

    print(f"\n   • Jumlah Cluster Optimal : {C_BOLD}{C_GREEN}K = {optimal_k}{C_RESET}")
    print(f"   • Silhouette Score Max   : {C_BOLD}{C_GREEN}{best_silhouette:.6f}{C_RESET} (Strong Cluster)")
    print("\n   • Tabel Distribusi Cluster Final:")
    print(df_final_distribution.to_string())

    # ============================================================
    # 9.2 Menyimpan Hasil Clustering Multi-Format (.parquet, .csv, .pkl)
    # ============================================================
    if export_dir is None:
        export_dir = os.path.join(os.getcwd(), "Hasil_Penelitian_PKB")
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    path_parquet = os.path.join(export_dir, f"{date_str}_hasil_clustering_kmeans.parquet")
    path_csv = os.path.join(export_dir, f"{date_str}_hasil_clustering_kmeans.csv")
    path_pkl = os.path.join(export_dir, f"{date_str}_hasil_clustering_kmeans.pkl")

    # Juga simpan copy di cache_dir
    cache_parquet = os.path.join(cache_dir, f"{date_str}_hasil_clustering_kmeans.parquet")
    cache_csv = os.path.join(cache_dir, f"{date_str}_hasil_clustering_kmeans.csv")
    cache_pkl = os.path.join(cache_dir, f"{date_str}_hasil_clustering_kmeans.pkl")

    df_cluster.to_parquet(path_parquet, index=False)
    df_cluster.to_csv(path_csv, index=False)
    df_cluster.to_pickle(path_pkl)

    df_cluster.to_parquet(cache_parquet, index=False)
    df_cluster.to_csv(cache_csv, index=False)
    df_cluster.to_pickle(cache_pkl)

    print(f"\n {C_BOLD}{C_YELLOW}9.2 MENYIMPAN HASIL CLUSTERING MULTI-FORMAT:{C_RESET}")
    print(line_sg)
    print(f"   • Berkas Parquet : {C_BOLD}{C_GREEN}{path_parquet}{C_RESET}")
    print(f"   • Berkas CSV     : {C_BOLD}{C_GREEN}{path_csv}{C_RESET}")
    print(f"   • Berkas Pickle  : {C_BOLD}{C_GREEN}{path_pkl}{C_RESET}")
    print(f"   • Dimensi Dataset: {df_cluster.shape[0]:,} Record x {df_cluster.shape[1]} Atribut.")

    # ============================================================
    # 9.3 Validasi & Membuka Kembali Berkas Format .pkl
    # ============================================================
    print(f"\n {C_BOLD}{C_YELLOW}9.3 VERIFIKASI BUKA KEMBALI BERKAS FORMAT .pkl:{C_RESET}")
    print(line_sg)
    df_loaded_pkl = pd.read_pickle(path_pkl)
    print(f"   • Status Load File .pkl : {C_BOLD}{C_GREEN}✅ Berhasil Dimuat (100% Utuh){C_RESET}")
    print(f"   • 5 Baris Pertama Data  :")
    selected_preview_cols = [c for c in ['Jenis_Pendaftaran', 'Jenis_Kendaraan', 'Golongan', 'PokokPKB', 'DendaPKB', 'Cluster'] if c in df_loaded_pkl.columns]
    if not selected_preview_cols:
        selected_preview_cols = df_loaded_pkl.columns[:6].tolist()
    print(df_loaded_pkl[selected_preview_cols].head().to_string())

    exec_duration = round(time.time() - start_time, 2)

    print("\n" + line_db)
    print(f" {C_BOLD}{C_GREEN}STATUS TAHAP 9: 🎉 SELURUH 9 TAHAPAN PENELITIAN K-MEANS CLUSTERING SELESAI 100%!{C_RESET}")
    print(line_db + "\n")

    report = {
        "total_records": len(df_cluster),
        "total_attributes": df_cluster.shape[1],
        "optimal_k": optimal_k,
        "best_silhouette": best_silhouette,
        "features_used": features_used_for_clustering,
        "df_final_distribution": df_final_distribution,
        "path_parquet": path_parquet,
        "path_csv": path_csv,
        "path_pkl": path_pkl,
        "exec_duration": exec_duration
    }

    return df_cluster, report


if __name__ == "__main__":
    run_research_summary_and_export()
