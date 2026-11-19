import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List, Optional

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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


def run_kmeans_and_evaluation(
    df_scaled_input: Optional[pd.DataFrame] = None,
    df_transform_input: Optional[pd.DataFrame] = None,
    sample_ratio: float = 0.30,
    random_state: int = 100,
    k_min: int = 2,
    k_max: int = 5
) -> Tuple[pd.DataFrame, KMeans, Dict[str, Any]]:
    """
    [TAHAP 6 & 7] PENAMBANGAN DATA (K-MEANS) & EVALUASI (SILHOUETTE SCORE) - METODOLOGI KANG YAYAT:
    6.1 Menyiapkan Data Eksperimen (Sampling 30%)
    6.2 Elbow Method (WCSS / Inertia K=2 s/d K=5)
    6.3 Pengujian K-Means Kandidat & Validasi Penurunan WCSS
    6.4 Persiapan K-Means Final
    7.1 Perbandingan Silhouette Score K=2, 3, 4, 5
    7.2 Penentuan K Optimal (Silhouette Max)
    7.3 Visualisasi Silhouette Score
    7.4a 2D Scatter Plot Hasil K-Means dengan PCA
    7.4b 3D Scatter Plot Hasil K-Means dengan PCA
    7.5 Silhouette Plot K Optimal
    7.6 K-Means Final (Fit 100% Seluruh Data)
    """
    start_time = time.time()
    W = 85
    line_db = C_CYAN + "═" * W + C_RESET
    line_sg = C_DIM + "─" * W + C_RESET

    print("\n" + line_db)
    print(f" {C_BOLD}{C_MAGENTA}🤖 [TAHAP 6 & 7] PENAMBANGAN DATA (K-MEANS) & EVALUASI SILHOUETTE{C_RESET}")
    print(f" {C_DIM}Tim Pengembang: Mang Ewok (Senior DS) & Kang Yayat | Fitur K-Means PKB{C_RESET}")
    print(line_db)

    # 1. Pemuatan Data dari Tahap 5
    cache_dir = os.path.join(os.getcwd(), ".cache_dataset")
    if df_scaled_input is None:
        scaled_cache_path = os.path.join(cache_dir, "transformed_scaled_dataset.parquet")
        if os.path.exists(scaled_cache_path):
            print(f" 📦 Memuat X_kmeans dari Cache Tahap 5: {C_DIM}{os.path.basename(scaled_cache_path)}{C_RESET}")
            X_kmeans = pd.read_parquet(scaled_cache_path)
        else:
            raise FileNotFoundError("Cache dataset hasil normalisasi (transformed_scaled_dataset.parquet) tidak ditemukan!")
    else:
        X_kmeans = df_scaled_input.copy()

    if df_transform_input is None:
        transform_cache_path = os.path.join(cache_dir, "preprocessed_dataset.parquet")
        if os.path.exists(transform_cache_path):
            df_transform = pd.read_parquet(transform_cache_path)
        else:
            df_transform = X_kmeans.copy()
    else:
        df_transform = df_transform_input.copy()

    # ============================================================
    # 6.1 Menyiapkan Data Eksperimen (Sampling 30%)
    # ============================================================
    sample_size = int(len(X_kmeans) * sample_ratio)
    if len(X_kmeans) > sample_size:
        X_sample = X_kmeans.sample(n=sample_size, random_state=random_state).copy()
    else:
        X_sample = X_kmeans.copy()

    pct_sample = (len(X_sample) / len(X_kmeans)) * 100
    print(f"\n {C_BOLD}{C_YELLOW}6.1 HASIL SAMPLING DATA EKSPERIMEN:{C_RESET}")
    print(line_sg)
    print(f"   • Jumlah data awal      : {len(X_kmeans):,} data")
    print(f"   • Jumlah data eksperimen: {len(X_sample):,} data ({pct_sample:.2f}%)")
    print(f"   • Jumlah atribut        : {X_sample.shape[1]}")
    print(f"   • Atribut yang digunakan : {C_CYAN}{', '.join(X_sample.columns)}{C_RESET}")

    # ============================================================
    # 6.2 Elbow Method & 6.3 Pengujian K-Means Kandidat
    # ============================================================
    k_range = range(k_min, k_max + 1)
    wcss = []
    models = {}

    for k in k_range:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            random_state=random_state,
            n_init=20
        )
        model.fit(X_sample)
        wcss.append(model.inertia_)
        models[k] = model

    df_wcss = pd.DataFrame({"K": list(k_range), "WCSS": wcss})

    print(f"\n {C_BOLD}{C_YELLOW}6.2 & 6.3 HASIL ELBOW METHOD & MODEL KANDIDAT:{C_RESET}")
    print(line_sg)
    for idx, row_w in df_wcss.iterrows():
        print(f"   • K = {int(row_w['K'])} | WCSS / Inertia = {row_w['WCSS']:>12,.2f}")

    # Validasi Penurunan WCSS
    print("\n   • Validasi Penurunan WCSS:")
    for i in range(len(wcss) - 1):
        k_now = list(k_range)[i]
        k_next = list(k_range)[i + 1]
        if wcss[i] >= wcss[i + 1]:
            print(f"     ✓ WCSS K={k_now} ({wcss[i]:,.0f}) >= WCSS K={k_next} ({wcss[i+1]:,.0f})")
        else:
            print(f"     ⚠️ PERINGATAN: WCSS K={k_now} < WCSS K={k_next}")

    # ============================================================
    # 7.1 & 7.2 EVALUASI SILHOUETTE SCORE & PENENTUAN K OPTIMAL
    # ============================================================
    evaluation_results = []
    for k, model in models.items():
        labels_sp = model.labels_
        score = silhouette_score(X_sample, labels_sp)
        evaluation_results.append({"K": k, "WCSS": model.inertia_, "Silhouette Score": score})

    df_evaluation = pd.DataFrame(evaluation_results).sort_values(by="K").reset_index(drop=True)
    best_result = df_evaluation.loc[df_evaluation["Silhouette Score"].idxmax()]

    optimal_k = int(best_result["K"])
    best_silhouette = float(best_result["Silhouette Score"])
    best_wcss = float(best_result["WCSS"])

    print(f"\n {C_BOLD}{C_YELLOW}7.1 & 7.2 HASIL EVALUASI & PENENTUAN K OPTIMAL:{C_RESET}")
    print(line_sg)
    print(df_evaluation.to_string(index=False))
    print(line_sg)
    print(f"   • {C_BOLD}K Optimal Terpilih   : {C_GREEN}K = {optimal_k}{C_RESET}")
    print(f"   • WCSS pada K Optimal  : {best_wcss:,.2f}")
    print(f"   • Silhouette Score Max : {C_BOLD}{C_GREEN}{best_silhouette:.6f}{C_RESET}")

    # Interpretasi Silhouette Score
    if 0.71 <= best_silhouette <= 1.0:
        interp = "Struktur cluster sangat kuat (Strong Cluster)"
    elif 0.51 <= best_silhouette < 0.71:
        interp = "Struktur cluster baik (Reasonable Cluster)"
    elif 0.26 <= best_silhouette < 0.51:
        interp = "Struktur cluster lemah (Weak Cluster)"
    else:
        interp = "Struktur cluster tidak terbentuk dengan baik (Wrong Cluster)"
    print(f"   • Interpretasi Clusterting: {C_BOLD}{C_CYAN}{interp}{C_RESET}")

    # ============================================================
    # 7.4a PCA 2D & 7.4b PCA 3D Information
    # ============================================================
    best_model_sample = models[optimal_k]
    labels_sample = best_model_sample.labels_

    pca_2d = PCA(n_components=2, random_state=random_state)
    X_pca_2d_arr = pca_2d.fit_transform(X_sample)
    exp_var_2d = pca_2d.explained_variance_ratio_ * 100
    df_pca_2d = pd.DataFrame(X_pca_2d_arr, columns=["PCA1", "PCA2"])
    df_pca_2d["Cluster"] = labels_sample
    centroids_pca_2d = pca_2d.transform(best_model_sample.cluster_centers_)

    pca_3d = PCA(n_components=3, random_state=random_state)
    X_pca_3d_arr = pca_3d.fit_transform(X_sample)
    exp_var_3d = pca_3d.explained_variance_ratio_ * 100
    df_pca_3d = pd.DataFrame(X_pca_3d_arr, columns=["PCA1", "PCA2", "PCA3"])
    df_pca_3d["Cluster"] = labels_sample
    centroids_pca_3d = pca_3d.transform(best_model_sample.cluster_centers_)

    print(f"\n {C_BOLD}{C_YELLOW}7.4 VISUALISASI PCA (2D & 3D):{C_RESET}")
    print(line_sg)
    print(f"   • Total Variansi PCA 2D : {exp_var_2d.sum():.2f}% (PCA1: {exp_var_2d[0]:.2f}%, PCA2: {exp_var_2d[1]:.2f}%)")
    print(f"   • Total Variansi PCA 3D : {exp_var_3d.sum():.2f}% (PCA1: {exp_var_3d[0]:.2f}%, PCA2: {exp_var_3d[1]:.2f}%, PCA3: {exp_var_3d[2]:.2f}%)")

    # ============================================================
    # 7.6 K-MEANS FINAL (FIT SELURUH DATA 100%)
    # ============================================================
    print(f"\n {C_BOLD}{C_YELLOW}7.6 K-MEANS FINAL BERDASARKAN K OPTIMAL (SELURUH DATA):{C_RESET}")
    print(line_sg)

    best_model_final = KMeans(
        n_clusters=optimal_k,
        init="k-means++",
        random_state=random_state,
        n_init=20
    )
    
    cluster_labels_final = best_model_final.fit_predict(X_kmeans)
    
    df_cluster = df_transform.copy()
    df_cluster["Cluster"] = cluster_labels_final
    X_kmeans_clustered = X_kmeans.copy()
    X_kmeans_clustered["Cluster"] = cluster_labels_final

    cluster_counts = pd.Series(cluster_labels_final).value_counts().sort_index()
    cluster_pcts = (cluster_counts / len(df_cluster)) * 100
    
    df_distribution = pd.DataFrame({
        "Jumlah Data": cluster_counts,
        "Persentase (%)": cluster_pcts.round(2)
    })
    print(df_distribution.to_string())

    # Save Clustered Data to Cache Parquet
    clustered_cache_path = os.path.join(cache_dir, "clustered_dataset_final.parquet")
    df_cluster.to_parquet(clustered_cache_path, index=False)
    
    exec_duration = round(time.time() - start_time, 2)

    print("\n" + line_db)
    print(f" {C_BOLD}{C_GREEN}STATUS TAHAP 6 & 7: ✅ PENAMBANGAN DATA & EVALUASI K-MEANS SELESAI 100%! SIAP KA TAHAP 8 (PROFILING){C_RESET}")
    print(line_db + "\n")

    report = {
        "sample_size": len(X_sample),
        "total_rows_initial": len(X_kmeans),
        "sample_pct": pct_sample,
        "df_wcss": df_wcss,
        "df_evaluation": df_evaluation,
        "optimal_k": optimal_k,
        "best_silhouette": best_silhouette,
        "best_wcss": best_wcss,
        "interpretation": interp,
        "exp_var_2d": exp_var_2d,
        "exp_var_3d": exp_var_3d,
        "df_pca_2d": df_pca_2d,
        "centroids_pca_2d": centroids_pca_2d,
        "df_pca_3d": df_pca_3d,
        "centroids_pca_3d": centroids_pca_3d,
        "df_distribution": df_distribution,
        "clustered_cache_path": clustered_cache_path,
        "exec_duration": exec_duration
    }

    return df_cluster, best_model_final, report


if __name__ == "__main__":
    run_kmeans_and_evaluation()
