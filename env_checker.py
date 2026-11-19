import os
import gc
import math
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import platform
import subprocess
import psutil
import multiprocessing
import shutil
import sklearn
import datetime
import importlib.metadata

from tqdm.auto import tqdm
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Menonaktifkan peringatan yang tidak perlu
warnings.filterwarnings("ignore")


def print_environment_dashboard():
    """
    [TAHAP 1] PERSIAPAN LINGKUNGAN - METODOLOGI PENELITIAN KANG YAYAT:
    Penelitian: Clustering Tingkat Kepatuhan Wajib Pajak Kendaraan Bermotor Menggunakan K-Means
    1. Informasi Python
    2. Sistem Operasi
    3. Informasi CPU
    4. Informasi RAM
    5. Informasi Storage
    6. Informasi GPU
    7. Versi Library Python
    8. Direktori Kerja
    9. Informasi Google Colab
    10. Informasi Linux
    """
    print("==================================================")
    print("= Lingkungan Google Colab / Python berhasil dipersiapkan. =")
    print("==================================================")

    print("=" * 70)
    print("SPESIFIKASI LINGKUNGAN PENELITIAN KANG YAYAT")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. INFORMASI PYTHON
    # ------------------------------------------------------------
    print("\n[1] INFORMASI PYTHON")
    print("-" * 70)
    print("Versi Python          :", sys.version)
    print("Executable Python     :", sys.executable)

    # ------------------------------------------------------------
    # 2. INFORMASI SISTEM OPERASI
    # ------------------------------------------------------------
    print("\n[2] SISTEM OPERASI")
    print("-" * 70)
    print("Sistem Operasi        :", platform.system())
    print("Versi OS              :", platform.version())
    print("Arsitektur            :", platform.machine())
    print("Platform              :", platform.platform())

    # ------------------------------------------------------------
    # 3. INFORMASI CPU
    # ------------------------------------------------------------
    print("\n[3] INFORMASI CPU")
    print("-" * 70)
    print("Processor              :", platform.processor() or "Generic Processor")
    print("Jumlah CPU Logical     :", multiprocessing.cpu_count())
    print("CPU Usage              :", psutil.cpu_percent(interval=1), "%")

    # ------------------------------------------------------------
    # 4. INFORMASI RAM
    # ------------------------------------------------------------
    print("\n[4] INFORMASI RAM")
    print("-" * 70)
    ram = psutil.virtual_memory()
    print("Total RAM              :", round(ram.total / (1024**3), 2), "GB")
    print("RAM Digunakan          :", round(ram.used / (1024**3), 2), "GB")
    print("RAM Tersedia           :", round(ram.available / (1024**3), 2), "GB")
    print("Persentase RAM         :", ram.percent, "%")

    # ------------------------------------------------------------
    # 5. INFORMASI STORAGE
    # ------------------------------------------------------------
    print("\n[5] INFORMASI STORAGE")
    print("-" * 70)
    cwd_drive = os.getcwd()
    disk = shutil.disk_usage(cwd_drive)
    print("Total Storage          :", round(disk.total / (1024**3), 2), "GB")
    print("Storage Digunakan      :", round(disk.used / (1024**3), 2), "GB")
    print("Storage Tersedia       :", round(disk.free / (1024**3), 2), "GB")
    print("Persentase Penggunaan  :", round((disk.used / disk.total) * 100, 2), "%")

    # ------------------------------------------------------------
    # 6. INFORMASI GPU
    # ------------------------------------------------------------
    print("\n[6] INFORMASI GPU")
    print("-" * 70)
    try:
        gpu_info = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode("utf-8")
        print(gpu_info)
    except Exception:
        print("GPU NVIDIA tidak tersedia atau GPU belum diaktifkan.")

    # ------------------------------------------------------------
    # 7. INFORMASI LIBRARY UTAMA
    # ------------------------------------------------------------
    print("\n[7] VERSI LIBRARY PYTHON")
    print("-" * 70)
    print("NumPy                 :", np.__version__)
    print("Pandas                :", pd.__version__)
    print("Scikit-learn          :", sklearn.__version__)
    print("Matplotlib            :", matplotlib.__version__)
    print("Seaborn               :", sns.__version__)
    print("Psutil                :", psutil.__version__)
    try:
        print("tqdm                  :", importlib.metadata.version('tqdm'))
    except Exception:
        print("tqdm                  :", "Terinstal")
    try:
        print("Scipy                 :", importlib.metadata.version('scipy'))
    except Exception:
        print("Scipy                 :", scipy.__version__)

    # ------------------------------------------------------------
    # 8. INFORMASI DIREKTORI KERJA
    # ------------------------------------------------------------
    print("\n[8] DIREKTORI KERJA")
    print("-" * 70)
    print("Working Directory     :", os.getcwd())

    # ------------------------------------------------------------
    # 9. INFORMASI ENVIRONMENT COLAB
    # ------------------------------------------------------------
    print("\n[9] INFORMASI GOOGLE COLAB")
    print("-" * 70)
    try:
        import google.colab
        print("Google Colab          : Aktif")
    except ImportError:
        print("Google Colab          : Tidak terdeteksi (Lokal/IDE Environment)")

    # ------------------------------------------------------------
    # 10. INFORMASI LINUX
    # ------------------------------------------------------------
    print("\n[10] INFORMASI LINUX")
    print("-" * 70)
    try:
        print(subprocess.check_output(["uname", "-a"]).decode("utf-8"))
    except Exception:
        print("Informasi Linux tidak tersedia (Windows/Non-Linux OS).")

    print("=" * 70)
    print("PEMERIKSAAN SPESIFIKASI SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    print_environment_dashboard()
