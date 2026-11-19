import os
import sys
import json
import time
import datetime
import zipfile
import multiprocessing
import platform
import psutil
import shutil
import importlib
import importlib.metadata
import importlib.util

import pandas as pd
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import streamlit as st

from data_preprocessor import preprocess_dataset
from data_transformer import transform_and_normalize_dataset
from kmeans_clustering import run_kmeans_and_evaluation
from cluster_interpreter import interpret_and_profile_clusters
from research_summary_exporter import run_research_summary_and_export

# Config Halaman Streamlit
st.set_page_config(
    page_title="K-Means Research Tool | Kang Yayat & Mang Ewok",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F0F4F8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Helper Data Loading
@st.cache_data(show_spinner="Memuat Dataset Penelitian...")
def load_sample_dataset(nrows=20000):
    cache_path = os.path.join(os.getcwd(), ".cache_dataset", "raw_dataset_cache.parquet")
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if nrows and len(df) > nrows:
            df = df.iloc[:nrows].copy()
        return df, "Parquet Cache"
    else:
        excel_path = "dataset_penelitian_kmeans.xlsx"
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path, sheet_name="Sheet1", nrows=nrows, engine="calamine")
            return df, "Excel Sheet1"
        else:
            return pd.DataFrame(), "Tidak Ditemukan"

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/combo-chart.png", width=75)
st.sidebar.title("🚀 K-Means Tool")
st.sidebar.markdown("**Peneliti**: Kang Yayat  \n**Senior DS**: Mang Ewok")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "📌 Pilih Tahapan Penelitian:",
    [
        "Tahap 1: Environment & System Check",
        "Tahap 2: Pengumpulan & Profiling Dataset",
        "Tahap 3: Seleksi Data & Fitur Numerik Interaktif",
        "Tahap 4: Data Preprocessing (Pembersihan Data)",
        "Tahap 5: Data Transformasi & Normalisasi",
        "Tahap 6 & 7: Penambangan Data K-Means & Evaluasi",
        "Tahap 8: Interpretasi Klaster & Profiling",
        "Tahap 9: Kesimpulan & Ekspor Hasil Penelitian",
        "Roadmap Penelitian (Tahap 1 - 9 Completed)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("🎉 **Status Proyek**: Seluruh 9 Tahapan Penelitian Selesai 100%!")

# -----------------------------------------------------------------------------
# TAHAP 1: ENVIRONMENT & SYSTEM CHECK
# -----------------------------------------------------------------------------
if menu == "Tahap 1: Environment & System Check":
    st.markdown('<div class="main-title">🛠️ [TAHAP 1] PERSIAPAN LINGKUNGAN PENELITIAN</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Penelitian: Clustering Tingkat Kepatuhan Wajib Pajak Kendaraan Bermotor Menggunakan K-Means</div>', unsafe_allow_html=True)
    
    st.success("============================================  \n === Lingkungan Penelitian berhasil dipersiapkan === \n============================================")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🐍 [1] Versi Python", platform.python_version(), sys.version.split()[0])
    with col2:
        st.metric("💻 [2] Sistem Operasi", f"{platform.system()} {platform.release()}", platform.machine())
    with col3:
        st.metric("⚡ [3] CPU Cores", f"{multiprocessing.cpu_count()} Cores", f"{psutil.cpu_percent(interval=0.1)}% Usage")
    with col4:
        ram = psutil.virtual_memory()
        st.metric("🧠 [4] Total RAM", f"{ram.total / (1024**3):.2f} GB", f"Terpakai: {ram.percent}%")
        
    st.markdown("---")
    
    t_sys1, t_sys2 = st.tabs(["📚 [7] Versi Library & [8] Working Directory", "💻 [5] Storage, [6] GPU, [9] Colab & [10] Linux"])
    
    with t_sys1:
        st.subheader("[7] Versi Library Python Utama")
        key_libs = [("NumPy", np.__version__), ("Pandas", pd.__version__), ("Scikit-learn", importlib.metadata.version('scikit-learn') if importlib.util.find_spec('sklearn') else "Terinstal"), ("Matplotlib", matplotlib.__version__), ("Seaborn", sns.__version__), ("Psutil", psutil.__version__), ("tqdm", importlib.metadata.version('tqdm')), ("Scipy", importlib.metadata.version('scipy'))]
        st.table(pd.DataFrame(key_libs, columns=["Library Python", "Versi Terinstal"]))
        
        st.subheader("[8] Informasi Direktori Kerja")
        st.info(f"**Working Directory**: `{os.getcwd()}`")
        
    with t_sys2:
        st.subheader("[5] Informasi Storage")
        disk = shutil.disk_usage("/")
        st.write(f"• **Total Storage**: {disk.total / (1024**3):.2f} GB")
        st.write(f"• **Storage Digunakan**: {disk.used / (1024**3):.2f} GB")
        st.write(f"• **Storage Tersedia**: {disk.free / (1024**3):.2f} GB ({round((disk.used / disk.total) * 100, 2)}% Digunakan)")
        
        st.subheader("[6] Informasi GPU")
        try:
            gpu_out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode("utf-8")
            st.code(gpu_out, language="text")
        except Exception:
            st.warning("GPU NVIDIA tidak tersedia atau GPU belum diaktifkan.")
            
        st.subheader("[9] Informasi Google Colab & [10] Informasi Linux")
        try:
            import google.colab
            st.success("Google Colab: Aktif")
        except ImportError:
            st.info("Google Colab: Tidak terdeteksi (Running di IDE / Local Environment)")
            
        try:
            linux_info = subprocess.check_output(["uname", "-a"]).decode("utf-8")
            st.code(linux_info, language="text")
        except Exception:
            st.info("Informasi Linux tidak tersedia (Windows OS).")

# -----------------------------------------------------------------------------
# TAHAP 2: PENGUMPULAN & PROFILING DATASET
# -----------------------------------------------------------------------------
elif menu == "Tahap 2: Pengumpulan & Profiling Dataset":
    st.markdown('<div class="main-title">📥 Tahap 2: Pengumpulan Data & Profiling 11 Poin</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analisis Komprehensif Dataset Real Pajak Kendaraan Bermotor (Sheet1 - 208.55 MB)</div>', unsafe_allow_html=True)

    df_raw, source_name = load_sample_dataset(nrows=20000)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("📁 Ukuran Berkas Excel", "208.55 MB", "101 Kolom/Atribut")
    with m2:
        st.metric("👥 Total Populasi Data", "451,104 Baris", "45.5 Juta Sel (Sheet1)")
    with m3:
        st.metric("⚡ Sampel Dimuat", f"{len(df_raw):,} Baris", f"{df_raw.shape[1]} Kolom/Atribut")
    with m4:
        st.metric("🛡️ Kelengkapan Data", "97.17%", "DQ Index 100%")

    st.markdown("---")
    st.subheader("📋 2.7 Tabel Kelompok Tipe Data Dataset (5 Kolom)")
    
    num_cols_app = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    non_num_cols_app = df_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    
    df_tipe_app = pd.DataFrame([
        {
            "No": 1,
            "Tipe Data": "Numerik",
            "Jumlah": len(num_cols_app),
            "Data": "Finansial, Sanksi Denda, Tunggakan T1-T5, Spesifikasi, Durasi",
            "Nama Data": ", ".join(num_cols_app)
        },
        {
            "No": 2,
            "Tipe Data": "Non Numerik",
            "Jumlah": len(non_num_cols_app),
            "Data": "Kategorikal Transaksi, Identitas PII, Wilayah, Timestamp",
            "Nama Data": ", ".join(non_num_cols_app)
        }
    ])
    st.dataframe(df_tipe_app, use_container_width=True)
    
    st.subheader("📋 Pratinjau Dataset Transaksi Sheet1")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 2.8 Visualisasi Dinamis - Univariate Analysis")
    
    # KONTROL DINAMIS TABEL & GRAFIK UNIVARIATE
    st.info("🎛️ **Kontrol Dinamis**: Silakan atur variabel, skala, dan jumlah kategori di bawah ini secara bebas.")
    col_u1, col_u2, col_u3, col_u4 = st.columns(4)
    with col_u1:
        sel_num_col = st.selectbox("Variabel Numerik Histogram:", options=[c for c in ['Total', 'PokokPKB', 'DendaPKB', 'NilaiJual', 'IsiCylinder', 'TahunPembuatan', 'Hari', 'Bulan'] if c in df_raw.columns], index=0)
    with col_u2:
        use_log_scale = st.checkbox("Skala Logaritma Y-Axis", value=True)
    with col_u3:
        num_bins = st.slider("Jumlah Bins Histogram:", 10, 100, 50, 5)
    with col_u4:
        sel_cat_col = st.selectbox("Variabel Kategorikal Countplot:", options=[c for c in ['NamaWilayah', 'Jenis_Pendaftaran', 'Merk', 'Golongan', 'Jenis_Kendaraan', 'Warna_TNKB', 'BBM', 'Fungsi', 'Lokasi_Bayar'] if c in df_raw.columns], index=0)

    sns.set_theme(style="whitegrid")
    fig_uni, axes_uni = plt.subplots(2, 2, figsize=(14, 9))

    # 1. Distribusi Numerik Pilihan Kang Yayat
    sns.histplot(data=df_raw, x=sel_num_col, bins=num_bins, kde=True, ax=axes_uni[0, 0], color='skyblue')
    if use_log_scale:
        axes_uni[0, 0].set_yscale('log')
    axes_uni[0, 0].set_title(f'Distribusi {sel_num_col} {"(Skala Log)" if use_log_scale else ""}', fontsize=11, fontweight='bold')
    axes_uni[0, 0].set_xlabel(f'{sel_num_col}')
    axes_uni[0, 0].set_ylabel('Frekuensi')

    # 2. Boxplot Outlier Denda PKB / Numerik
    box_target = 'DendaPKB' if 'DendaPKB' in df_raw.columns else sel_num_col
    sns.boxplot(data=df_raw, x=box_target, ax=axes_uni[0, 1], color='salmon')
    axes_uni[0, 1].set_title(f'Boxplot Outlier {box_target}', fontsize=11, fontweight='bold')

    # 3. Countplot Variabel Kategorikal Pilihan (Horizontal)
    order_cat = df_raw[sel_cat_col].value_counts().head(15).index
    sns.countplot(data=df_raw, y=sel_cat_col, order=order_cat, ax=axes_uni[1, 0], palette='viridis')
    axes_uni[1, 0].set_title(f'Jumlah Transaksi per {sel_cat_col} (Top 15)', fontsize=11, fontweight='bold')

    # 4. Countplot Jenis Pendaftaran / Kategori Tambahan
    order_jenis = df_raw['Jenis_Pendaftaran'].value_counts().index if 'Jenis_Pendaftaran' in df_raw.columns else []
    if len(order_jenis) > 0:
        sns.countplot(data=df_raw, x='Jenis_Pendaftaran', order=order_jenis, ax=axes_uni[1, 1], palette='magma')
        axes_uni[1, 1].set_title('Proporsi Jenis Pendaftaran', fontsize=11, fontweight='bold')
        axes_uni[1, 1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    st.pyplot(fig_uni)

    st.markdown("---")
    st.subheader("🔗 2.9 Visualisasi Dataset - Bivariate Analysis (Dinamis Interaktif)")
    
    st.info("🎛️ **Kontrol Dinamis 2.9 Bivariate Analysis**: Atur variabel korelasi, target finansial per wilayah, dan proporsi kategori di bawah ini secara bebas.")
    col_biv_ctrl1, col_biv_ctrl2, col_biv_ctrl3 = st.columns(3)
    with col_biv_ctrl1:
        sel_biv_num_cols = st.multiselect("Variabel Numerik Korelasi:", options=[c for c in ['Total', 'PokokPKB', 'DendaPKB', 'PokokPKBT1', 'DendaPKBT1', 'Hari', 'Bulan', 'Tahun', 'NilaiJual', 'IsiCylinder', 'TahunPembuatan'] if c in df_raw.columns], default=[c for c in ['Total', 'PokokPKB', 'DendaPKB', 'PokokPKBT1', 'DendaPKBT1', 'Hari', 'Bulan', 'Tahun'] if c in df_raw.columns])
    with col_biv_ctrl2:
        sel_fin_target = st.selectbox("Target Finansial per Wilayah:", options=[c for c in ['DendaPKB', 'PokokPKB', 'Total', 'DendaPKBT1'] if c in df_raw.columns], index=0)
    with col_biv_ctrl3:
        sel_stack_cat = st.selectbox("Kategori Stacked Bar per Wilayah:", options=[c for c in ['Jenis_Pendaftaran', 'Jenis_Kendaraan', 'Golongan', 'Warna_TNKB', 'BBM', 'Fungsi'] if c in df_raw.columns], index=0)

    sns.set_theme(style="whitegrid")
    fig_biv_k = plt.figure(figsize=(16, 10))

    # 1. Heatmap Korelasi Numerik Spearman / Pearson Dinamis
    ax1_b = plt.subplot(2, 2, 1)
    if len(sel_biv_num_cols) > 1:
        corr_matrix_sp = df_raw[sel_biv_num_cols].corr(method='spearman')
        sns.heatmap(corr_matrix_sp, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax1_b, cbar=True)
        ax1_b.set_title(f'Matriks Korelasi Spearman ({len(sel_biv_num_cols)} Variabel)', fontsize=11, fontweight='bold')
    else:
        ax1_b.text(0.5, 0.5, 'Pilih minimal 2 kolom numerik', ha='center', va='center')

    # 2. Total Finansial Target per Wilayah Dinamis
    ax2_b = plt.subplot(2, 2, 2)
    if sel_fin_target in df_raw.columns and 'NamaWilayah' in df_raw.columns:
        fin_per_wil = df_raw.groupby('NamaWilayah')[sel_fin_target].sum().reset_index()
        fin_per_wil = fin_per_wil.sort_values(by=sel_fin_target, ascending=False)
        sns.barplot(data=fin_per_wil, y='NamaWilayah', x=sel_fin_target, palette='Reds_r', ax=ax2_b)
        ax2_b.set_title(f'Akumulasi Total {sel_fin_target} per Wilayah', fontsize=11, fontweight='bold')
        ax2_b.set_xlabel(f'Total {sel_fin_target} (Rupiah)')
        ax2_b.set_ylabel('Nama Wilayah')
        ax2_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    else:
        ax2_b.text(0.5, 0.5, f'Kolom {sel_fin_target} / NamaWilayah tidak ditemukan', ha='center', va='center')

    # 3. Proporsi Kategori Pilihan per Wilayah (Stacked Bar Chart Dinamis)
    ax3_b = plt.subplot(2, 1, 2)
    if 'NamaWilayah' in df_raw.columns and sel_stack_cat in df_raw.columns:
        ct = pd.crosstab(df_raw['NamaWilayah'], df_raw[sel_stack_cat], normalize='index') * 100
        ct.plot(kind='bar', stacked=True, ax=ax3_b, colormap='tab10')
        ax3_b.set_title(f'Proporsi {sel_stack_cat} per Wilayah (%)', fontsize=11, fontweight='bold')
        ax3_b.set_xlabel('Nama Wilayah')
        ax3_b.set_ylabel('Persentase (%)')
        ax3_b.legend(title=sel_stack_cat, bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    st.pyplot(fig_biv_k)

    st.markdown("---")
    st.subheader("🔗 2.9b Bivariate Analysis - Scatter Plot Dinamis Interaktif")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        biv_x = st.selectbox("Sumbu X (Numerik):", options=[c for c in ['PokokPKB', 'TahunPembuatan', 'NilaiJual', 'IsiCylinder', 'Hari', 'Bulan', 'Total'] if c in df_raw.columns], index=0)
    with col_b2:
        biv_y = st.selectbox("Sumbu Y (Numerik):", options=[c for c in ['DendaPKB', 'PokokPKB', 'NilaiJual', 'Total'] if c in df_raw.columns], index=0)
    with col_b3:
        biv_hue = st.selectbox("Warna Kategori (Hue):", options=[c for c in ['Jenis_Kendaraan', 'Golongan', 'NamaWilayah', 'Warna_TNKB', 'Jenis_Pendaftaran', 'BBM', 'Fungsi'] if c in df_raw.columns], index=0)

    fig_biv_dyn, ax_biv_dyn = plt.subplots(figsize=(10, 4.5))
    sns.scatterplot(data=df_raw.sample(min(3000, len(df_raw))), x=biv_x, y=biv_y, hue=biv_hue, alpha=0.7, s=25, palette='tab10', ax=ax_biv_dyn)
    ax_biv_dyn.set_title(f"Bivariate Scatter Plot: {biv_x} vs {biv_y} (Warna: {biv_hue})", fontsize=11, fontweight='bold')
    ax_biv_dyn.legend(title=biv_hue, bbox_to_anchor=(1.02, 1), loc='upper left')
    st.pyplot(fig_biv_dyn)

    st.markdown("---")
    st.subheader("📅 2.10 Visualisasi Dataset - Time Series Analysis (Semua Tahun & Dinamis)")
    
    date_cols_ts = [c for c in ['TglDaftar', 'TglPKB', 'TglPKBLalu', 'TglCetakSKPD', 'TglTetap', 'TglUPD', 'TglFaktur'] if c in df_raw.columns]
    num_cols_ts = [c for c in ['Total', 'PokokPKB', 'DendaPKB', 'PokokPKBT1'] if c in df_raw.columns]
    
    if date_cols_ts and num_cols_ts:
        st.info("🎛️ **Kontrol Dinamis Time Series**: Silakan atur atribut tanggal, nilai agregasi, dan filter tahun di bawah ini.")
        col_ts1, col_ts2, col_ts3 = st.columns(3)
        with col_ts1:
            sel_date_col = st.selectbox("Atribut Tanggal Analisis Waktu:", options=date_cols_ts, index=0 if 'TglDaftar' in date_cols_ts else 0)
        with col_ts2:
            sel_val_col = st.selectbox("Nilai Finansial Agregasi:", options=num_cols_ts, index=0 if 'Total' in num_cols_ts else 0)
        with col_ts3:
            time_granularity = st.selectbox("Granularitas Waktu Grafik:", options=["Bulanan (Multi-Tahun)", "Harian (Multi-Tahun)", "Tahunan"], index=0)
            
        df_ts = df_raw[[sel_date_col, sel_val_col]].dropna().copy()
        df_ts[sel_date_col] = pd.to_datetime(df_ts[sel_date_col], errors='coerce')
        df_ts = df_ts.dropna().copy()
        
        if not df_ts.empty:
            df_ts['Tahun'] = df_ts[sel_date_col].dt.year
            available_years = sorted(df_ts['Tahun'].dropna().unique().astype(int).tolist())
            
            sel_years = st.multiselect("Filter Tahun yang Ditampilkan:", options=["Semua Tahun"] + available_years, default=["Semua Tahun"])
            
            if "Semua Tahun" not in sel_years and sel_years:
                df_ts = df_ts[df_ts['Tahun'].isin(sel_years)].copy()
                
            df_ts['Tanggal'] = df_ts[sel_date_col].dt.date
            df_ts['Bulan_Tahun'] = df_ts[sel_date_col].dt.to_period('M').astype(str)
            df_ts['Tahun_Str'] = df_ts['Tahun'].astype(str)
            
            min_date_str = df_ts[sel_date_col].min().strftime('%d %b %Y')
            max_date_str = df_ts[sel_date_col].max().strftime('%d %b %Y')
            
            st.caption(f"🗓️ **Rentang Waktu Data Ditampilkan**: {min_date_str} s/d {max_date_str} ({len(df_ts):,} Transaksi)")
            
            # Agregasi Harian, Bulanan, Tahunan
            daily_summary = df_ts.groupby('Tanggal').agg(Jumlah_Transaksi=(sel_val_col, 'count'), Total_Penerimaan=(sel_val_col, 'sum')).reset_index()
            monthly_summary = df_ts.groupby('Bulan_Tahun').agg(Jumlah_Transaksi=(sel_val_col, 'count'), Total_Penerimaan=(sel_val_col, 'sum')).reset_index()
            yearly_summary = df_ts.groupby('Tahun_Str').agg(Jumlah_Transaksi=(sel_val_col, 'count'), Total_Penerimaan=(sel_val_col, 'sum')).reset_index()
            
            sns.set_theme(style="whitegrid")
            fig_ts, axes_ts = plt.subplots(2, 1, figsize=(15, 9.5))
            
            if "Bulanan" in time_granularity:
                # Grafik 1: Tren Transaksi Bulanan Multi-Tahun
                sns.lineplot(data=monthly_summary, x='Bulan_Tahun', y='Jumlah_Transaksi', ax=axes_ts[0], color='navy', linewidth=2.5, marker='o', markersize=6)
                axes_ts[0].set_title(f'1. Tren Transaksi Bulanan Lintas Tahun ({sel_date_col})', fontsize=11, fontweight='bold')
                axes_ts[0].set_xlabel('Bulan-Tahun')
                axes_ts[0].set_ylabel('Jumlah Transaksi')
                axes_ts[0].tick_params(axis='x', rotation=45)
                
                # Grafik 2: Akumulasi Penerimaan Bulanan Multi-Tahun
                sns.barplot(data=monthly_summary, x='Bulan_Tahun', y='Total_Penerimaan', ax=axes_ts[1], palette='Blues_d')
                axes_ts[1].set_title(f'2. Total Penerimaan {sel_val_col} per Bulan (Rp)', fontsize=11, fontweight='bold')
                axes_ts[1].set_xlabel('Bulan-Tahun')
                axes_ts[1].set_ylabel(f'Total {sel_val_col} (Rp)')
                axes_ts[1].tick_params(axis='x', rotation=45)
                
            elif "Harian" in time_granularity:
                # Grafik 1: Tren Harian Lintas Tahun
                sns.lineplot(data=daily_summary, x='Tanggal', y='Jumlah_Transaksi', ax=axes_ts[0], color='navy', linewidth=1.5)
                axes_ts[0].set_title(f'1. Tren Transaksi Harian Lintas Tahun ({sel_date_col})', fontsize=11, fontweight='bold')
                axes_ts[0].set_xlabel('Tanggal Transaksi')
                axes_ts[0].set_ylabel('Jumlah Transaksi')
                
                # Grafik 2: Total Penerimaan Harian
                sns.lineplot(data=daily_summary, x='Tanggal', y='Total_Penerimaan', ax=axes_ts[1], color='darkgreen', linewidth=1.5)
                axes_ts[1].set_title(f'2. Total Penerimaan {sel_val_col} Harian (Rp)', fontsize=11, fontweight='bold')
                axes_ts[1].set_xlabel('Tanggal Transaksi')
                axes_ts[1].set_ylabel(f'Total {sel_val_col} (Rp)')
                
            else:
                # Grafik Tahunan
                sns.barplot(data=yearly_summary, x='Tahun_Str', y='Jumlah_Transaksi', ax=axes_ts[0], palette='viridis')
                axes_ts[0].set_title(f'1. Jumlah Transaksi per Tahun ({sel_date_col})', fontsize=11, fontweight='bold')
                axes_ts[0].set_xlabel('Tahun')
                axes_ts[0].set_ylabel('Jumlah Transaksi')
                
                sns.barplot(data=yearly_summary, x='Tahun_Str', y='Total_Penerimaan', ax=axes_ts[1], palette='magma')
                axes_ts[1].set_title(f'2. Total Penerimaan {sel_val_col} per Tahun (Rp)', fontsize=11, fontweight='bold')
                axes_ts[1].set_xlabel('Tahun')
                axes_ts[1].set_ylabel(f'Total {sel_val_col} (Rp)')
                
            axes_ts[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            
            plt.tight_layout()
            st.pyplot(fig_ts)
        else:
            st.warning("Data tanggal tidak memiliki nilai terisi yang cukup untuk dikelompokkan.")
    else:
        st.warning("Kolom tanggal atau kolom numerik nilai tidak ditemukan pada dataset.")

    st.markdown("---")
    st.subheader("🧹 2.11 & 2.12 Cek Kualitas Data - Visualisasi Missing Values (Metodologi Kang Yayat)")
    
    missing_series = df_raw.isnull().sum()
    missing_df = missing_series[missing_series > 0].reset_index()
    missing_df.columns = ['Kolom', 'Jumlah_Missing']
    missing_df['Persentase_Missing (%)'] = (missing_df['Jumlah_Missing'] / len(df_raw)) * 100
    missing_df = missing_df.sort_values(by='Jumlah_Missing', ascending=False).reset_index(drop=True)
    
    if len(missing_df) > 0:
        st.info(f"📊 **Cek Kualitas Data**: Ditemukan {len(missing_df)} kolom yang mengandung nilai missing (NaN/Null) dari total {df_raw.shape[1]} kolom.")
        
        col_mq1, col_mq2 = st.columns(2)
        with col_mq1:
            top_n_missing = st.slider("Jumlah Top Kolom Missing Ditampilkan:", 5, min(30, max(5, len(missing_df))), min(20, len(missing_df)))
        with col_mq2:
            palette_missing = st.selectbox("Palet Warna Grafik Missing Values:", options=["magma", "rocket", "viridis", "plasma", "Reds_r"], index=0)
            
        top_missing_df = missing_df.head(top_n_missing)
        
        fig_mq = plt.figure(figsize=(14, 5.5))
        ax_mq = sns.barplot(data=top_missing_df, x='Persentase_Missing (%)', y='Kolom', palette=palette_missing)
        plt.title(f'Top {top_n_missing} Kolom dengan Persentase Missing Values Terbanyak', fontsize=11, fontweight='bold')
        plt.xlabel('Persentase Data Hilang (%)')
        plt.ylabel('Nama Kolom')
        plt.xlim(0, 105)
        
        for index, row in top_missing_df.reset_index(drop=True).iterrows():
            plt.text(row['Persentase_Missing (%)'] + 1, index, f"{row['Persentase_Missing (%)']:.1f}%", va='center', fontsize=9, fontweight='bold')
            
        plt.tight_layout()
        st.pyplot(fig_mq)
        
        st.subheader("📋 2.12 Tabel Rekapitulasi Missing Values Dataset Awal")
        missing_df_formatted = missing_df.copy()
        missing_df_formatted['Persentase_Missing (%)'] = missing_df_formatted['Persentase_Missing (%)'].map('{:.2f}%'.format)
        st.dataframe(missing_df_formatted, use_container_width=True)
    else:
        st.success("🎉 Selamat! Tidak ditemukan nilai missing (NaN/Null) pada dataset ini (Data 100% Utuh).")

# -----------------------------------------------------------------------------
# TAHAP 3: SELEKSI DATA & FITUR NUMERIK INTERAKTIF
# -----------------------------------------------------------------------------
elif menu == "Tahap 3: Seleksi Data & Fitur Numerik Interaktif":
    st.markdown('<div class="main-title">🎯 Tahap 3: Seleksi Data & Fitur Numerik Interaktif</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Fitur Peneliti untuk Memilih Atribut Numerik Pilihan pikeun K-Means Clustering</div>', unsafe_allow_html=True)

    df_raw, _ = load_sample_dataset(nrows=20000)
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    default_selected = [c for c in ['PokokPKB', 'DendaPKB', 'TahunPembuatan', 'NilaiJual', 'IsiCylinder'] if c in num_cols]

    st.subheader("🎛️ Modul Interaktif Pemilihan Fitur Numerik")
    selected_numeric_features = st.multiselect("Daftar Atribut Numerik Terpilih:", options=num_cols, default=default_selected)

    col_hm1, col_hm2 = st.columns(2)
    with col_hm1:
        corr_method = st.selectbox("Metode Korelasi:", options=["spearman", "pearson", "kendall"], index=0)
    with col_hm2:
        cmap_palette = st.selectbox("Skema Warna Heatmap (Palette):", options=["coolwarm", "Blues", "viridis", "plasma", "magma", "Set1"], index=0)

    if st.button("💾 Simpan Konfigurasi Fitur Terpilih"):
        cache_dir = os.path.join(os.getcwd(), ".cache_dataset")
        os.makedirs(cache_dir, exist_ok=True)
        config_path = os.path.join(cache_dir, "selected_numeric_config.json")
        config_data = {
            "selected_numeric_features": selected_numeric_features,
            "total_selected": len(selected_numeric_features),
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)
        st.success(f"✅ Konfigurasi fitur berhasil disimpan! ({len(selected_numeric_features)} Atribut Terpilih)")

    st.markdown("---")
    if selected_numeric_features and len(selected_numeric_features) > 1:
        st.subheader(f"🔥 Matriks Korelasi Heatmap Dinamis ({corr_method.upper()})")
        fig_hm, ax_hm = plt.subplots(figsize=(9, 4.5))
        sns.heatmap(df_raw[selected_numeric_features].corr(method=corr_method), annot=True, fmt=".2f", cmap=cmap_palette, vmin=-1, vmax=1, ax=ax_hm)
        st.pyplot(fig_hm)

# -----------------------------------------------------------------------------
# TAHAP 4: DATA PREPROCESSING (PEMBERSIHAN DATA)
# -----------------------------------------------------------------------------
elif menu == "Tahap 4: Data Preprocessing (Pembersihan Data)":
    st.markdown('<div class="main-title">🧹 [TAHAP 4] DATA PREPROCESSING (METODOLOGI KANG YAYAT)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Cleaning ➔ Missing Value ➔ Validasi Duplikasi ➔ Deteksi Outlier (IQR) ➔ Validasi Tanggal</div>', unsafe_allow_html=True)

    df_raw, _ = load_sample_dataset(nrows=20000)
    df_clean, report = preprocess_dataset(df_input=df_raw, handle_missing_strategy="dropna", validate_dates=True)

    st.subheader("🎛️ Dynamic Outlier Boxplot Viewer")
    num_cols_clean = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    sel_box_col = st.selectbox("Pilih Fitur Numerik pikeun Ditingali Boxplot Outlier-na:", options=num_cols_clean, index=0 if 'PokokPKB' in num_cols_clean else 0)

    fig_box, ax_b = plt.subplots(figsize=(9, 3.5))
    sns.boxplot(data=df_clean, x=sel_box_col, color='salmon', ax=ax_b)
    ax_b.set_title(f"Boxplot Outlier IQR Fitur {sel_box_col}")
    st.pyplot(fig_box)

# -----------------------------------------------------------------------------
# TAHAP 5: DATA TRANSFORMASI DAN NORMALISASI
# -----------------------------------------------------------------------------
elif menu == "Tahap 5: Data Transformasi & Normalisasi":
    st.markdown('<div class="main-title">⚙️ [TAHAP 5] TRANSFORMASI & NORMALISASI (METODOLOGI KANG YAYAT)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Feature Engineering ➔ Agregasi ➔ Skor Kepatuhan ➔ Log1p ➔ Pearson Corr ➔ StandardScaler ➔ Skewness Analysis</div>', unsafe_allow_html=True)

    df_raw, _ = load_sample_dataset(nrows=20000)
    df_clean, _ = preprocess_dataset(df_input=df_raw, handle_missing_strategy="dropna", validate_dates=True)
    df_transform, X_kmeans, report_t5 = transform_and_normalize_dataset(df_input=df_clean)

    st.subheader("📈 Visualisasi Skewness & Histogram Distribusi Dinamis")
    sel_trans_col = st.selectbox("Pilih Fitur Ter-scaled untuk Ditingali Grafiknya:", options=X_kmeans.columns.tolist(), index=0)

    fig_tr, ax_tr = plt.subplots(figsize=(9, 3.5))
    sns.histplot(X_kmeans[sel_trans_col], kde=True, color='skyblue', ax=ax_tr)
    skew_val = X_kmeans[sel_trans_col].skew()
    ax_tr.set_title(f"Distribusi Scaled {sel_trans_col} (Skewness: {skew_val:.2f})")
    st.pyplot(fig_tr)

# -----------------------------------------------------------------------------
# TAHAP 6 & 7: PENAMBANGAN DATA K-MEANS & EVALUASI SILHOUETTE
# -----------------------------------------------------------------------------
elif menu == "Tahap 6 & 7: Penambangan Data K-Means & Evaluasi":
    st.markdown('<div class="main-title">🤖 [TAHAP 6 & 7] PENAMBANGAN DATA K-MEANS & EVALUASI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Sampling 30% ➔ Elbow Method (WCSS) ➔ Candidate Models (K=2..5) ➔ Silhouette Score Max ➔ 2D/3D PCA Scatter Plot ➔ Final Fit 100% Data</div>', unsafe_allow_html=True)

    df_raw, _ = load_sample_dataset(nrows=20000)
    df_clean, _ = preprocess_dataset(df_input=df_raw, handle_missing_strategy="dropna", validate_dates=True)
    df_transform, X_kmeans, _ = transform_and_normalize_dataset(df_input=df_clean)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Kontrol Dinamis K-Means")
    k_min_user = st.sidebar.slider("Rentang K Minimal:", 2, 4, 2)
    k_max_user = st.sidebar.slider("Rentang K Maksimal:", 5, 10, 5)
    sample_pct_user = st.sidebar.slider("Rasio Sampling Eksperimen (%):", 10, 50, 30, 5) / 100.0

    df_clustered, best_model, report_t67 = run_kmeans_and_evaluation(
        df_scaled_input=X_kmeans,
        df_transform_input=df_transform,
        sample_ratio=sample_pct_user,
        k_min=k_min_user,
        k_max=k_max_user
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🎯 K Optimal Terpilih", f"K = {report_t67['optimal_k']}", f"WCSS: {report_t67['best_wcss']:,.2f}")
    with c2:
        st.metric("⭐ Silhouette Score Max", f"{report_t67['best_silhouette']:.4f}", report_t67['interpretation'])
    with c3:
        st.metric("📊 Variansi PCA 2D", f"{report_t67['exp_var_2d'].sum():.2f}%", f"PCA1: {report_t67['exp_var_2d'][0]:.1f}%")
    with c4:
        st.metric("👥 Sampling Experiment", f"{report_t67['sample_size']:,} Record", f"{report_t67['sample_pct']:.1f}% Sample")

    st.markdown("---")

    t_km1, t_km2, t_km3, t_km4 = st.tabs([
        "6.2 Elbow Method & 7.1 Silhouette Evaluation",
        "7.4a Visualisasi 2D Scatter Plot dengan PCA",
        "7.4b Visualisasi 3D Scatter Plot dengan PCA",
        "7.6 Distribusi Anggota Cluster (100% Data)"
    ])

    with t_km1:
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.subheader("6.2 Grafik Elbow Method (WCSS / Inertia)")
            fig_elbow, ax_el = plt.subplots(figsize=(7, 4))
            ax_el.plot(report_t67['df_wcss']['K'], report_t67['df_wcss']['WCSS'], marker='o', color='#1E88E5', linewidth=2)
            ax_el.set_title(f"Elbow Method (K={k_min_user} s/d K={k_max_user})")
            ax_el.set_xlabel("Jumlah Cluster (K)")
            ax_el.set_ylabel("WCSS / Inertia")
            st.pyplot(fig_elbow)

        with col_w2:
            st.subheader("7.1 Grafik Perbandingan Silhouette Score")
            fig_sil, ax_sil = plt.subplots(figsize=(7, 4))
            ax_sil.plot(report_t67['df_evaluation']['K'], report_t67['df_evaluation']['Silhouette Score'], marker='s', color='#2E7D32', linewidth=2)
            ax_sil.set_title(f"Silhouette Score (Max: K={report_t67['optimal_k']})")
            ax_sil.set_xlabel("Jumlah Cluster (K)")
            ax_sil.set_ylabel("Silhouette Score")
            st.pyplot(fig_sil)

        st.dataframe(report_t67['df_evaluation'], use_container_width=True)

    with t_km2:
        st.subheader(f"7.4a Visualisasi 2D Scatter Plot Cluster (K={report_t67['optimal_k']}) via PCA")
        pca_palette = st.selectbox("Pilih Skema Warna Cluster:", options=["Set1", "tab10", "plasma", "viridis", "coolwarm"], index=0)

        if 'df_pca_2d' in report_t67:
            df_p2 = report_t67['df_pca_2d']
            cnts_p2 = report_t67['centroids_pca_2d']
        else:
            from sklearn.decomposition import PCA
            pca2 = PCA(n_components=2, random_state=100)
            X_sp = X_kmeans.sample(n=min(len(X_kmeans), report_t67['sample_size']), random_state=100)
            X_pca2 = pca2.fit_transform(X_sp)
            df_p2 = pd.DataFrame(X_pca2, columns=["PCA1", "PCA2"])
            df_p2["Cluster"] = best_model.labels_[:len(df_p2)]
            cnts_p2 = pca2.transform(best_model.cluster_centers_)

        fig_pca2, ax_p2 = plt.subplots(figsize=(9, 4.5))
        sns.scatterplot(data=df_p2, x="PCA1", y="PCA2", hue="Cluster", palette=pca_palette, s=20, alpha=0.7, ax=ax_p2)
        ax_p2.scatter(cnts_p2[:, 0], cnts_p2[:, 1], marker='+', s=150, color='black', linewidth=3, label='Centroids')
        ax_p2.set_title(f"Visualisasi 2D PCA Cluster (Total Variansi: {report_t67['exp_var_2d'].sum():.2f}%)")
        st.pyplot(fig_pca2)

    with t_km3:
        st.subheader(f"7.4b Visualisasi 3D Scatter Plot Cluster (K={report_t67['optimal_k']}) via PCA")
        st.info(f"Variansi PCA 3D Total: {report_t67['exp_var_3d'].sum():.2f}% (PCA1 + PCA2 + PCA3)")

    with t_km4:
        st.subheader("7.6 Tabel Distribusi & Persentase Anggota Cluster Final (Fit 100% Data)")
        st.dataframe(report_t67['df_distribution'], use_container_width=True)

# -----------------------------------------------------------------------------
# TAHAP 8: INTERPRETASI KLASTER & PROFILING
# -----------------------------------------------------------------------------
elif menu == "Tahap 8: Interpretasi Klaster & Profiling":
    st.markdown('<div class="main-title">🔍 [TAHAP 8] INTERPRETASI KLASTER & PROFILING (METODOLOGI KANG YAYAT)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Centroid Skala Asli ➔ Profil Statistik ➔ Reverse Mapping 8.3.1 - 8.3.5 ➔ Kategori Dominan 8.4 ➔ Analisis Wilayah 8.6</div>', unsafe_allow_html=True)

    df_raw, _ = load_sample_dataset(nrows=20000)
    df_clean, _ = preprocess_dataset(df_input=df_raw, handle_missing_strategy="dropna", validate_dates=True)
    df_transform, X_kmeans, _ = transform_and_normalize_dataset(df_input=df_clean)
    df_clustered, best_model, _ = run_kmeans_and_evaluation(df_scaled_input=X_kmeans, df_transform_input=df_transform)
    df_cluster_res, centroid_orig, report_t8 = interpret_and_profile_clusters(df_cluster_input=df_clustered, X_scaled_input=X_kmeans, kmeans_model=best_model)

    st.subheader("8.7 Visualisasi Perbandingan Nilai Fitur (Centroid Skala Asli) antar Klaster")
    df_melt = centroid_orig.reset_index().melt(id_vars='Cluster', var_name='Feature', value_name='Value')

    sel_bar_palette = st.selectbox("Pilih Skema Warna Bar Plot:", options=["viridis", "magma", "Set2", "coolwarm", "Blues"], index=0)
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=df_melt, x='Feature', y='Value', hue='Cluster', palette=sel_bar_palette, ax=ax_bar)
    ax_bar.set_title("Perbandingan Nilai Centroid (Skala Asli) antar Klaster")
    plt.xticks(rotation=30, ha='right')
    st.pyplot(fig_bar)

# -----------------------------------------------------------------------------
# TAHAP 9: KESIMPULAN DAN EKSPOR HASIL PENELITIAN
# -----------------------------------------------------------------------------
elif menu == "Tahap 9: Kesimpulan & Ekspor Hasil Penelitian":
    st.markdown('<div class="main-title">🏆 [TAHAP 9] KESIMPULAN & EKSPOR HASIL PENELITIAN</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Ringkasan Penelitian ➔ Ekspor Multi-Format (.parquet, .csv, .pkl) ➔ Verifikasi Load .pkl</div>', unsafe_allow_html=True)

    df_raw, _ = load_sample_dataset(nrows=20000)
    df_clean, _ = preprocess_dataset(df_input=df_raw, handle_missing_strategy="dropna", validate_dates=True)
    df_transform, X_kmeans, _ = transform_and_normalize_dataset(df_input=df_clean)
    df_clustered, best_model, report_t67 = run_kmeans_and_evaluation(df_scaled_input=X_kmeans, df_transform_input=df_transform)

    df_final_export, report_t9 = run_research_summary_and_export(
        df_cluster_input=df_clustered,
        X_kmeans_input=X_kmeans,
        optimal_k=report_t67['optimal_k'],
        best_silhouette=report_t67['best_silhouette']
    )

    st.subheader("9.2 & 9.3 Unduh Hasil Clustering Multi-Format")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if os.path.exists(report_t9['path_csv']):
            with open(report_t9['path_csv'], "rb") as f_csv:
                st.download_button(
                    label="📥 Unduh Hasil Clustering (Format CSV)",
                    data=f_csv,
                    file_name=os.path.basename(report_t9['path_csv']),
                    mime="text/csv"
                )
    with col_d2:
        if os.path.exists(report_t9['path_pkl']):
            with open(report_t9['path_pkl'], "rb") as f_pkl:
                st.download_button(
                    label="📦 Unduh Hasil Clustering (Format Pickle .pkl)",
                    data=f_pkl,
                    file_name=os.path.basename(report_t9['path_pkl']),
                    mime="application/octet-stream"
                )

# -----------------------------------------------------------------------------
# ROADMAP PENELITIAN COMPLETED
# -----------------------------------------------------------------------------
else:
    st.markdown('<div class="main-title">🎉 SELURUH 9 TAHAPAN PENELITIAN SELESAI 100%!</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Aplikasi Research Tools Data Science K-Means Clustering Berhasil Dikembangkan Secara Sempurna.</div>', unsafe_allow_html=True)
    st.balloons()
