import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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


def interpret_and_profile_clusters(
    df_cluster_input: Optional[pd.DataFrame] = None,
    X_scaled_input: Optional[pd.DataFrame] = None,
    kmeans_model: Optional[KMeans] = None,
    scaler: Optional[StandardScaler] = None,
    file_path: str = "dataset_penelitian_kmeans.xlsx"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    [TAHAP 8] INTERPRETASI KLASTER - METODOLOGI PENELITIAN KANG YAYAT:
    8.1 Centroid dalam Skala Standardisasi
    8.2 Centroid dalam Skala Asli (Inverse Transform)
    8.3 Profil Statistik Setiap Cluster (count, mean, median)
    8.3.1 s/d 8.3.5 Validasi Tabulasi Silang (Reverse Mapping) Khusus Klaster 0 s/d 4
    8.4 Persiapan & Analisis Proporsi Atribut Kategorikal per Cluster
    8.4.1 s/d 8.4.5 Heatmap, Kategori Dominan, & Ringkasan Karakteristik Kategorikal
    8.5 Distribusi Jumlah Anggota Cluster
    8.6 & 8.8 Analisis Cluster Berdasarkan Wilayah (NamaWilayah vs Cluster)
    8.7 Visualisasi Perbandingan Fitur antar Klaster (Centroid Skala Asli)
    """
    start_time = time.time()
    W = 85
    line_db = C_CYAN + "═" * W + C_RESET
    line_sg = C_DIM + "─" * W + C_RESET

    print("\n" + line_db)
    print(f" {C_BOLD}{C_MAGENTA}🔍 [TAHAP 8] INTERPRETASI KLASTER (PROFILING & KARAKTERISTIK KLASTER){C_RESET}")
    print(f" {C_DIM}Tim Pengembang: Mang Ewok (Senior DS) & Kang Yayat | Algoritma: K-Means{C_RESET}")
    print(line_db)

    cache_dir = os.path.join(os.getcwd(), ".cache_dataset")

    # 1. Pemuatan Dataset Terklasterisasi ti Tahap 7 / Cache
    if df_cluster_input is None:
        clustered_path = os.path.join(cache_dir, "clustered_dataset_final.parquet")
        if os.path.exists(clustered_path):
            print(f" 📦 Memuat df_cluster dari Cache Tahap 7: {C_DIM}{os.path.basename(clustered_path)}{C_RESET}")
            df_cluster = pd.read_parquet(clustered_path)
        else:
            raise FileNotFoundError("Cache dataset terklasterisasi (clustered_dataset_final.parquet) tidak ditemukan!")
    else:
        df_cluster = df_cluster_input.copy()

    if X_scaled_input is None:
        scaled_path = os.path.join(cache_dir, "transformed_scaled_dataset.parquet")
        if os.path.exists(scaled_path):
            X_kmeans = pd.read_parquet(scaled_path)
        else:
            X_kmeans = df_cluster.select_dtypes(include=[np.number]).copy()
    else:
        X_kmeans = X_scaled_input.copy()

    features = [c for c in ["PokokPKB", "DendaPKB", "DurasiTunggakan", "SkorTNKB", "SkorGolonganKendaraan", "SkorFungsiKendaraan"] if c in df_cluster.columns or c in X_kmeans.columns]
    
    # 2. Fit Scaler jika scaler tidak diberikan
    if scaler is None:
        scaler = StandardScaler()
        df_num_feat = df_cluster[features].copy() if all(c in df_cluster.columns for c in features) else X_kmeans[features].copy()
        scaler.fit(df_num_feat)

    # 3. K-Means Centroid Extraction
    best_k = df_cluster["Cluster"].nunique()
    if kmeans_model is None:
        kmeans_model = KMeans(n_clusters=best_k, init="k-means++", random_state=100, n_init=20)
        df_num_for_fit = df_cluster[features].copy()
        X_for_fit = scaler.transform(df_num_for_fit)
        kmeans_model.fit(X_for_fit)

    # ============================================================
    # 8.1 Centroid dalam Skala Standardisasi
    # ============================================================
    centroid_scaled = pd.DataFrame(kmeans_model.cluster_centers_, columns=features)
    centroid_scaled.index.name = "Cluster"

    print(f"\n {C_BOLD}{C_YELLOW}8.1 CENTROID DALAM SKALA STANDARDISASI (μ = 0, σ = 1):{C_RESET}")
    print(line_sg)
    print(centroid_scaled.round(4).to_string())

    # ============================================================
    # 8.2 Centroid dalam Skala Asli (Inverse Transform)
    # ============================================================
    centroid_original = pd.DataFrame(scaler.inverse_transform(kmeans_model.cluster_centers_), columns=features)
    centroid_original.index.name = "Cluster"

    print(f"\n {C_BOLD}{C_YELLOW}8.2 CENTROID DALAM SKALA ASLI (INVERSE TRANSFORM):{C_RESET}")
    print(line_sg)
    print(centroid_original.round(2).to_string())

    # ============================================================
    # 8.3 Profil Statistik Setiap Cluster (count, mean, median)
    # ============================================================
    cluster_profile = df_cluster.groupby("Cluster")[features].agg(["count", "mean", "median"])

    print(f"\n {C_BOLD}{C_YELLOW}8.3 PROFIL STATISTIK SETIAP CLUSTER (COUNT, MEAN, MEDIAN):{C_RESET}")
    print(line_sg)
    print(cluster_profile.head(10).to_string())

    # ============================================================
    # 8.3.1 s/d 8.3.5 Validasi Tabulasi Silang (Reverse Mapping)
    # ============================================================
    print(f"\n {C_BOLD}{C_YELLOW}8.3.1 - 8.3.5 VALIDASI TABULASI SILANG (REVERSE MAPPING) KLASTER 0 s/d 4:{C_RESET}")
    print(line_sg)
    
    reverse_mapping_results = {}
    for c_id in sorted(df_cluster['Cluster'].unique()):
        klaster_sub = df_cluster[df_cluster['Cluster'] == c_id]
        print(f"   • Klaster {c_id}: {len(klaster_sub):,} kendaraan")
        
        tnkb_dist = (klaster_sub['Warna_TNKB'].value_counts(normalize=True) * 100).round(2) if 'Warna_TNKB' in klaster_sub.columns else pd.Series()
        fungsi_dist = (klaster_sub['Fungsi'].value_counts(normalize=True) * 100).round(2) if 'Fungsi' in klaster_sub.columns else pd.Series()
        gol_dist = (klaster_sub['Golongan'].value_counts(normalize=True) * 100).round(2) if 'Golongan' in klaster_sub.columns else pd.Series()
        
        reverse_mapping_results[c_id] = {
            "populasi": len(klaster_sub),
            "tnkb_dist": tnkb_dist,
            "fungsi_dist": fungsi_dist,
            "gol_dist": gol_dist
        }
        top_tnkb = tnkb_dist.index[0] if not tnkb_dist.empty else "N/A"
        top_fungsi = fungsi_dist.index[0] if not fungsi_dist.empty else "N/A"
        print(f"     - Top TNKB: '{top_tnkb}' ({tnkb_dist.iloc[0] if not tnkb_dist.empty else 0}%) | Top Fungsi: '{top_fungsi}' ({fungsi_dist.iloc[0] if not fungsi_dist.empty else 0}%)")

    # ============================================================
    # 8.4 Persiapan & Analisis Proporsi Atribut Kategorikal per Cluster
    # ============================================================
    categorical_features = ["Jenis_Pendaftaran", "Jenis_Kendaraan", "Warna_TNKB", "Golongan", "Fungsi", "Lokasi_Bayar", "NamaWilayah"]
    available_cat_features = [c for c in categorical_features if c in df_cluster.columns]

    print(f"\n {C_BOLD}{C_YELLOW}8.4 PEMERIKSAAN ATRIBUT KATEGORIKAL ({len(available_cat_features)} ATRIBUT TERSEDIA):{C_RESET}")
    print(line_sg)
    print(f"   • Atribut Tersedia: {C_GREEN}{', '.join(available_cat_features)}{C_RESET}")

    # ============================================================
    # 8.4.3 Identifikasi Kategori Dominan pada Setiap Cluster
    # ============================================================
    dominant_category_results = []
    for feat in available_cat_features:
        crosstab = pd.crosstab(df_cluster["Cluster"], df_cluster[feat])
        pct_table = (crosstab.div(crosstab.sum(axis=1), axis=0) * 100)
        for cluster in pct_table.index:
            dom_cat = pct_table.loc[cluster].idxmax()
            dom_pct = pct_table.loc[cluster].max()
            dominant_category_results.append({
                "Atribut": feat,
                "Cluster": cluster,
                "Kategori Dominan": dom_cat,
                "Proporsi (%)": round(dom_pct, 2)
            })

    df_dominant_category = pd.DataFrame(dominant_category_results)
    print(f"\n {C_BOLD}{C_YELLOW}8.4.3 KATEGORI DOMINAN PADA SETIAP CLUSTER:{C_RESET}")
    print(line_sg)
    print(df_dominant_category.head(15).to_string(index=False))

    # ============================================================
    # 8.6 & 8.8 Analisis Cluster Berdasarkan Wilayah
    # ============================================================
    wilayah_summary = pd.DataFrame()
    if "NamaWilayah" in df_cluster.columns:
        wilayah_cluster = pd.crosstab(df_cluster["NamaWilayah"], df_cluster["Cluster"])
        wilayah_cluster["Total"] = wilayah_cluster.sum(axis=1)
        if 1 in wilayah_cluster.columns:
            wilayah_cluster["Persentase_Cluster_1"] = (wilayah_cluster[1] / wilayah_cluster["Total"] * 100).round(2)
            wilayah_cluster = wilayah_cluster.sort_values("Persentase_Cluster_1", ascending=False)
        wilayah_summary = wilayah_cluster.copy()

    exec_duration = round(time.time() - start_time, 2)

    print("\n" + line_db)
    print(f" {C_BOLD}{C_GREEN}STATUS TAHAP 8: ✅ INTERPRETASI KLASTER SELESAI 100%! SIAP KA TAHAP 9 (KESIMPULAN){C_RESET}")
    print(line_db + "\n")

    report = {
        "centroid_scaled": centroid_scaled,
        "centroid_original": centroid_original,
        "cluster_profile": cluster_profile,
        "reverse_mapping_results": reverse_mapping_results,
        "available_cat_features": available_cat_features,
        "df_dominant_category": df_dominant_category,
        "wilayah_summary": wilayah_summary,
        "exec_duration": exec_duration
    }

    return df_cluster, centroid_original, report


if __name__ == "__main__":
    interpret_and_profile_clusters()
