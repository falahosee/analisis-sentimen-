import streamlit as st
import joblib
import pandas as pd
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sentimen AI Pro", layout="wide", page_icon="📈")

# --- 1. SISTEM LOGIN ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.markdown("<h1 style='text-align: center;'>🔐 Akses Analisis Sentimen</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="admin@gmail.com")
            password = st.text_input("Password", type="password", placeholder="tugasakhir")
            submit = st.form_submit_button("Login Sekarang")
            
            if submit:
                if email == "admin@gmail.com" and password == "tugasakhir":
                    st.session_state.logged_in = True
                    st.success("Login Berhasil!")
                    st.rerun()
                else:
                    st.error("Email atau Password salah!")

# --- 2. DASHBOARD UTAMA ---
def main_dashboard():
    # Load Model (PKL)
    @st.cache_resource
    def load_model():
        try:
            models = joblib.load("semua_model.pkl")
            tfidf = joblib.load("vectorizer.pkl")
            return models, tfidf
        except Exception as e:
            return None, None

    models, tfidf = load_model()
    
    if models is None:
        st.error("❌ File .pkl tidak ditemukan! Pastikan 'semua_model.pkl' dan 'vectorizer.pkl' ada di folder.")
        return

    # Sidebar
    st.sidebar.title("🛠️ Menu")
    st.sidebar.success(f"User Aktif: admin@gmail.com")
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🤖 AI Sentiment Analyzer & Data Processor")
    st.markdown("---")

    # --- FITUR 1: DATA SELECTION & UPLOAD (VERSI FIX SCRAPPING) ---
    st.subheader("📂 Step 1: Pilih Sumber Data")
    uploaded_file = st.file_uploader("Upload file CSV ulasan di sini", type=["csv"], key="main_uploader")
    
    df_aktif = None

    def read_safe_csv(file_source):
        # Membaca file sebagai teks mentah untuk menangani baris yang berantakan
        raw_data = file_source.read().decode('latin-1')
        lines = raw_data.splitlines()
        
        # Logika pembersihan: Ambil baris yang bukan header 'ulasan'
        clean_reviews = []
        for line in lines:
            l = line.strip()
            if not l or l.lower() == 'ulasan':
                continue
            clean_reviews.append(l)
        
        # Paksa menjadi satu kolom bernama 'ulasan' agar mesin bisa baca
        df = pd.DataFrame(clean_reviews, columns=['ulasan'])
        return df

    if uploaded_file is not None:
        try:
            df_aktif = read_safe_csv(uploaded_file)
            st.success(f"✅ File Berhasil Diunggah! ({len(df_aktif)} ulasan terbaca)")
        except Exception as e:
            st.error(f"Gagal membaca file upload: {e}")
            
    elif os.path.exists("ulasan_tokopedia_full.csv"):
        try:
            with open("ulasan_tokopedia_full.csv", 'rb') as f:
                df_aktif = read_safe_csv(f)
            st.info("ℹ️ Menggunakan data internal: ulasan_tokopedia_full.csv")
        except Exception as e:
            st.error(f"Gagal membaca file internal: {e}")
    else:
        st.warning("⚠️ Belum ada data. Silakan upload file CSV ulasan.")

    # --- FITUR 2: ANALISIS AI ---
    if df_aktif is not None:
        st.markdown("---")
        st.subheader("📊 Step 2: Analisis AI & Perbandingan 3 Model")

        # UI Pemilihan Kolom (Otomatis akan muncul 'ulasan')
        kolom_analisis = st.selectbox(
            "Pilih Kolom Teks yang Ingin Dianalisis:", 
            options=df_aktif.columns,
            index=0
        )

        if st.button("🚀 Jalankan Analisis Perbandingan"):
            with st.spinner('AI sedang memproses 3 model...'):
                for nama_model, model_obj in models.items():
                    df_aktif[f'Sentimen_{nama_model}'] = df_aktif[kolom_analisis].apply(
                        lambda x: model_obj.predict(tfidf.transform([str(x).lower()]))[0]
                    )
            
            st.success("✅ Analisis Selesai!")

            # Tabel Hasil
            kolom_tampil = [kolom_analisis] + [f'Sentimen_{m}' for m in models.keys()]
            st.dataframe(df_aktif[kolom_tampil], width="stretch")

            # Visualisasi 3 Grafik Berjejer
            st.markdown("### 📈 Distribusi Sentimen")
            c1, c2, c3 = st.columns(3)
            chart_cols = [c1, c2, c3]

            for i, name in enumerate(models.keys()):
                with chart_cols[i]:
                    st.write(f"**{name}**")
                    counts = df_aktif[f'Sentimen_{name}'].value_counts()
                    st.bar_chart(counts)

            # Fitur Download
            st.divider()
            csv_data = df_aktif.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Hasil (.csv)", csv_data, "hasil_sentimen.csv", "text/csv")
            st.balloons()

# --- KONTROL ALUR ---
if not st.session_state.logged_in:
    login_page()
else:
    main_dashboard()