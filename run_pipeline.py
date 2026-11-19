import os
import sys
import time
import datetime
import pandas as pd
import numpy as np

from env_checker import print_environment_dashboard
from data_collection_dashboard import run_full_data_collection_analysis
from data_selector import select_features_for_kmeans
from data_preprocessor import preprocess_dataset
from data_transformer import transform_and_normalize_dataset
from kmeans_clustering import run_kmeans_and_evaluation
from cluster_interpreter import interpret_and_profile_clusters
from research_summary_exporter import run_research_summary_and_export

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


def run_complete_research_pipeline(
    excel_file_path: str = "dataset_penelitian_kmeans.xlsx",
    sheet_name: str = "Sheet1",
    nrows: int = 20000
) -> Dict[str, Any]:
    """
    Fungsi Otomatisasi End-to-End Pipeline 9 Tahapan Penelitian Data Science.
    Bisa menerima dataset Excel BARU apapun secara fleksibel.
    
    :param excel_file_path: Path lokasi file Excel dataset baru.
    :param sheet_name: Nama sheet dataset utama.
    :param nrows: Jumlah baris data yang diproses.
    :return: Dictionary Ringkasan Eksekusi Pipeline
    """
    pipeline_start_time = time.time()
    W = 85
    line_db = C_CYAN + "═" * W + C_RESET
    line_sg = C_DIM + "─" * W + C_RESET

    print("\n" + line_db)
    print(f" {C_BOLD}{C_MAGENTA}🚀 PIPELINE EKSEKUSI OTOMATIS 9 TAHAPAN PENELITIAN K-MEANS CLUSTERING{C_RESET}")
    print(f" {C_DIM}Tim Pengembang: Mang Ewok (Senior DS) & Kang Yayat | Dataset Input: {os.path.basename(excel_file_path)}{C_RESET}")
    print(line_db)

    # -------------------------------------------------------------------------
    # TAHAP 1: PERSIPAN ENVIRONMENT
    # -------------------------------------------------------------------------
    print(f"\n {C_BOLD}{C_YELLOW}▶ TAHAP 1: PERSIPAN ENVIRONMENT & SYSTEM CHECK{C_RESET}")
    print_environment_dashboard()

    # -------------------------------------------------------------------------
    # TAHAP 2: PENGUMPULAN DATA & PROFILING 2.1 s/d 2.12
    # -------------------------------------------------------------------------
    print(f"\n {C_BOLD}{C_YELLOW}▶ TAHAP 2: PENGUMPULAN DATA & PROFILING (2.1 s/d 2.12){C_RESET}")
    df_raw, df_optimasi, report_t2 = run_full_data_collection_analysis(
        file_path=excel_file_path,
        sheet_name=sheet_name,
        nrows=nrows
    )

    # -------------------------------------------------------------------------
    # TAHAP 3: SELEKSI DATA 3.1 s/d 3.4
    # -------------------------------------------------------------------------
    print(f"\n {C_BOLD}{C_YELLOW}▶ TAHAP 3: SELEKSI DATA (3.1 s/d 3.4){C_RESET}")
    df_selected, report_t3 = select_features_for_kmeans(
        df_input=df_optimasi,
        file_path=excel_file_path,
        nrows=nrows
    )

    # -------------------------------------------------------------------------
    # TAHAP 4: DATA PREPROCESSING 4.1 s/d 4.8
    # -------------------------------------------------------------------------
    print(f"\n {C_BOLD}{C_YELLOW}▶ TAHAP 4: DATA PREPROCESSING (4.1 s/d 4.8){C_RESET}")
    df_clean, report_t4 = preprocess_dataset(
        df_input=df_selected,
        file_path=excel_file_path,
        handle_missing_strategy="dropna",
        validate_dates=True
    )

    # -------------------------------------------------------------------------
    # TAHAP 5: DATA TRANSFORMASI & NORMALISASI 5.1 s/d 5.10
    # -------------------------------------------------------------------------
    print(f"\n {C_BOLD}{C_YELLOW}▶ TAHAP 5: DATA TRANSFORMASI & NORMALISASI (5.1 s/d 5.10){C_RESET}")
    df_transform, X_kmeans, report_t5 = transform_and_normalize_dataset(
        df_input=df_clean,
        file_path=excel_file_path
    )

    # -------------------------------------------------------------------------
    # TAHAP 6 & 7: PENAMBANGAN DATA K-MEANS & EVALUASI SILHOUETTE
    # -------------------------------------------------------------------------
    print(f"\n {C_BOLD}{C_YELLOW}▶ TAHAP 6 & 7: PENAMBANGAN DATA K-MEANS & EVALUASI SILHOUETTE{C_RESET}")
    df_clustered, best_model, report_t67 = run_kmeans_and_evaluation(
        df_scaled_input=X_kmeans,
        df_transform_input=df_transform,
        sample_ratio=0.30,
        random_state=100
    )

    # -------------------------------------------------------------------------
    # TAHAP 8: INTERPRETASI KLASTER & PROFILING 8.1 s/d 8.8
    # -------------------------------------------------------------------------
    print(f"\n {C_BOLD}{C_YELLOW}▶ TAHAP 8: INTERPRETASI KLASTER & PROFILING (8.1 s/d 8.8){C_RESET}")
    df_cluster_res, centroid_orig, report_t8 = interpret_and_profile_clusters(
        df_cluster_input=df_clustered,
        X_scaled_input=X_kmeans,
        kmeans_model=best_model
    )

    # -------------------------------------------------------------------------
    # TAHAP 9: KESIMPULAN & EKSPOR MULTI-FORMAT 9.1 s/d 9.3
    # -------------------------------------------------------------------------
    print(f"\n {C_BOLD}{C_YELLOW}▶ TAHAP 9: KESIMPULAN & EKSPOR MULTI-FORMAT (9.1 s/d 9.3){C_RESET}")
    df_final_export, report_t9 = run_research_summary_and_export(
        df_cluster_input=df_clustered,
        X_kmeans_input=X_kmeans,
        optimal_k=report_t67['optimal_k'],
        best_silhouette=report_t67['best_silhouette']
    )

    total_pipeline_time = round(time.time() - pipeline_start_time, 2)

    print("\n" + line_db)
    print(f" {C_BOLD}{C_GREEN}🎉 VERIFIKASI ALUR PIPELINE PIPELINE 9 TAHAPAN SELESAI 100% SUKSES!{C_RESET}")
    print(f"   • Dataset Input          : {os.path.basename(excel_file_path)}")
    print(f"   • Total Data Berhasil    : {len(df_final_export):,} Record x {df_final_export.shape[1]} Atribut")
    print(f"   • K Optimal              : K = {report_t67['optimal_k']}")
    print(f"   • Silhouette Score       : {report_t67['best_silhouette']:.6f} ({report_t67['interpretation']})")
    print(f"   • Total Waktu Eksekusi   : {total_pipeline_time} detik")
    print(line_db + "\n")

    return {
        "dataset_file": excel_file_path,
        "total_records": len(df_final_export),
        "total_attributes": df_final_export.shape[1],
        "optimal_k": report_t67['optimal_k'],
        "best_silhouette": report_t67['best_silhouette'],
        "pipeline_duration_sec": total_pipeline_time
    }


if __name__ == "__main__":
    file_to_test = "dataset_penelitian_kmeans.xlsx"
    if len(sys.argv) > 1:
        file_to_test = sys.argv[1]
    run_complete_research_pipeline(excel_file_path=file_to_test)
