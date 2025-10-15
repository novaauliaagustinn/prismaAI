import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import joblib
import os
import base64

# --- Fungsi untuk men-set background ---
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: rgba(0,0,0,0);
    }}

    /* --- Style tambahan untuk logo dan teks --- */
    .logo-container {{
        display: flex;
        align-items: center;
        gap: 20px;
        position: absolute;
        top: 15px;
        left: 15px;
        z-index: 9999;
    }}
    h1, h2, h3, h4, h5, h6, p, label, div[data-testid="stMarkdownContainer"] {{
        color: #133E87; /* warna dari gambar */
        font-weight: 600;
    }}
    div[data-testid="stForm"] * {{
        color: white !important; /* teks dalam box tetap putih */
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# --- Konfigurasi halaman dan background ---
st.set_page_config(page_title="Klastering Pelanggan", layout="centered")
set_background("background.png")  # ganti dengan nama file background kamu

# --- Tambahkan dua logo di kiri atas (PLN dulu, baru Danantara) ---
try:
    pln_logo = base64.b64encode(open("pln.png", "rb").read()).decode()
    danantara_logo = base64.b64encode(open("danatara.png", "rb").read()).decode()

    st.markdown(
        f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{pln_logo}" width="80">
            <img src="data:image/png;base64,{danantara_logo}" width="80">
        </div>
        """,
        unsafe_allow_html=True
    )
except Exception as e:
    st.warning("⚠️ Logo PLN atau Danantara tidak ditemukan di folder aplikasi.")

# --- Judul utama ---
st.title("⚡ Klastering Data Pelanggan PLN UP 3 Surabaya Barat")

# --- Inisialisasi session_state ---
for key in ["idpel_new", "nama_new", "penyulang_new", "kwh_new", "jn_new"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# --- Input manual ---
st.header("✏️ Input Data Pelanggan Baru")
with st.form("manual_input", clear_on_submit=True):
    idpel_new = st.text_input("ID_PELANGGAN", value=st.session_state.idpel_new, key="idpel_input")
    nama_new = st.text_input("NAMA", value=st.session_state.nama_new, key="nama_input")

    if os.path.exists("labelencoder_penyulang.pkl"):
        le = joblib.load("labelencoder_penyulang.pkl")
        penyulang_options = sorted(le.classes_)
        penyulang_new = st.selectbox("PENYULANG", penyulang_options, key="penyulang_input")
    else:
        st.warning("⚠️ Model penyulang belum tersedia. Upload CSV dulu untuk melatih model.")
        penyulang_new = None

    kwh_new = st.text_input("KWH (gunakan koma, tanpa titik)", value=st.session_state.kwh_new, key="kwh_input")
    jn_new = st.text_input("JN (gunakan koma, tanpa titik)", value=st.session_state.jn_new, key="jn_input")

    submit_manual = st.form_submit_button("Prediksi Cluster")

# --- Upload CSV ---
uploaded_file = st.file_uploader("📁 Unggah file CSV pelanggan", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📋 Data Awal", df.head())

    try:
        df_klaster = df[['PENYULANG', 'KWH', 'JN']].copy()
        le = LabelEncoder()
        df_klaster['PENYULANG'] = le.fit_transform(df_klaster['PENYULANG'])
        df_klaster['KWH'] = df_klaster['KWH'].astype(str).str.replace(',', '').astype(float)
        df_klaster['JN'] = df_klaster['JN'].astype(str).str.replace(',', '').astype(float)

        kmeans = KMeans(n_clusters=4, random_state=8)
        df['Cluster'] = kmeans.fit_predict(df_klaster) + 1

        joblib.dump(kmeans, "model_kmeans.pkl")
        joblib.dump(le, "labelencoder_penyulang.pkl")

        st.success("✅ Klastering selesai dan model disimpan!")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh hasil CSV", csv, "hasil_klaster.csv", "text/csv")

    except Exception as e:
        st.error("❌ Kesalahan saat memproses data:")
        st.exception(e)

# --- Prediksi manual ---
if submit_manual:
    try:
        if not (os.path.exists("model_kmeans.pkl") and os.path.exists("labelencoder_penyulang.pkl")):
            st.error("❌ Model belum tersedia. Silakan upload CSV untuk melatih model terlebih dahulu.")
        else:
            kmeans = joblib.load("model_kmeans.pkl")
            le = joblib.load("labelencoder_penyulang.pkl")

            try:
                kwh_float = float(kwh_new.replace(",", "").strip())
                jn_float = float(jn_new.replace(",", "").strip())
            except ValueError:
                st.error("⚠️ Format KWH/JN tidak valid.")
                st.stop()

            if penyulang_new not in le.classes_:
                st.error(f"⚠️ PENYULANG '{penyulang_new}' belum pernah muncul di data training.")
            else:
                penyulang_enc = le.transform([penyulang_new])
                X_new = [[penyulang_enc[0], kwh_float, jn_float]]
                cluster_pred = int(kmeans.predict(X_new)[0]) + 1

                st.success(f"✅ Data pelanggan '{nama_new}' masuk Cluster {cluster_pred}")

                if "hasil_prediksi" not in st.session_state:
                    st.session_state.hasil_prediksi = pd.DataFrame(
                        columns=["ID_PELANGGAN", "NAMA", "PENYULANG", "KWH", "JN", "CLUSTER"]
                    )

                hasil_baru = pd.DataFrame([{
                    "ID_PELANGGAN": idpel_new,
                    "NAMA": nama_new,
                    "PENYULANG": penyulang_new,
                    "KWH": kwh_new,
                    "JN": jn_new,
                    "CLUSTER": cluster_pred
                }])

                st.session_state.hasil_prediksi = pd.concat(
                    [st.session_state.hasil_prediksi, hasil_baru],
                    ignore_index=True
                )

                for key in ["idpel_new", "nama_new", "penyulang_new", "kwh_new", "jn_new"]:
                    st.session_state[key] = ""

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat prediksi manual.")
        st.exception(e)

# --- Cari pelanggan ---
st.header("🔍 Cari Pelanggan Berdasarkan IDPEL atau NAMA")
search_query = st.text_input("Masukkan IDPEL atau NAMA:")

data_gabungan = pd.DataFrame()

if "df" in locals():
    data_gabungan = pd.concat(
        [data_gabungan, df[["ID_PELANGGAN", "NAMA", "PENYULANG", "KWH", "JN", "Cluster"]]], ignore_index=True
    )

if "hasil_prediksi" in st.session_state and not st.session_state.hasil_prediksi.empty:
    data_gabungan = pd.concat(
        [data_gabungan, st.session_state.hasil_prediksi.rename(columns={"CLUSTER": "Cluster"})], ignore_index=True
    )

if data_gabungan.empty:
    st.info("ℹ️ Belum ada data. Upload CSV atau tambahkan data manual terlebih dahulu.")
else:
    if search_query:
        hasil_cari = data_gabungan[
            data_gabungan["ID_PELANGGAN"].astype(str).str.contains(search_query, case=False, na=False) |
            data_gabungan["NAMA"].astype(str).str.contains(search_query, case=False, na=False)
        ]
        if not hasil_cari.empty:
            st.success(f"Ditemukan {len(hasil_cari)} hasil:")
            st.dataframe(hasil_cari)
        else:
            st.warning("⚠️ Tidak ditemukan pelanggan dengan kata kunci tersebut.")
    else:
        st.write("Ketik IDPEL atau NAMA untuk mencari pelanggan.")

# --- Tabel hasil manual ---
if "hasil_prediksi" in st.session_state and not st.session_state.hasil_prediksi.empty:
    st.subheader("📋 Hasil Prediksi Manual (Semua Input)")
    st.dataframe(st.session_state.hasil_prediksi)