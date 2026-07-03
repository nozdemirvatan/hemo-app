"""
MDS Admin Paneli
================
Erişim: http://localhost:8501/admin
Sadece yetkili kullanıcı içindir.
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="MDS Admin Paneli", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)
# ── Şifre koruması ─────────────────────────────────────────────────────
ADMIN_SIFRE = "MDS2026"  # ← buraya kendi şifreni yaz

if "admin_giris" not in st.session_state:
    st.session_state.admin_giris = False

if not st.session_state.admin_giris:
    st.title("🔒 Admin Girişi")
    sifre = st.text_input("Şifre", type="password")
    if st.button("Giriş"):
        if sifre == ADMIN_SIFRE:
            st.session_state.admin_giris = True
            st.rerun()
        else:
            st.error("Hatalı şifre.")
    st.stop()

# ── Admin paneli ────────────────────────────────────────────────────────
st.title("🛠️ MDS Admin Paneli")

LOG_FILE = "tahmin_log.csv"

if not os.path.exists(LOG_FILE):
    st.info("Henüz hiç tahmin kaydı yok.")
    st.stop()

df = pd.read_csv(LOG_FILE)

# ── Özet istatistik ────────────────────────────────────────────────────
st.subheader("Özet")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam Tahmin", len(df))
col2.metric("Pozitif (Tarama ≥0.50)", int((df["risk_skoru"] >= 0.50).sum()))
col3.metric("Pozitif (Dengeli ≥0.60)", int((df["risk_skoru"] >= 0.60).sum()))
col4.metric("Tam Mod Kullanımı", int((df["mod"] == "D_tam").sum()))

st.divider()

# ── Filtreler ──────────────────────────────────────────────────────────
st.subheader("Tahmin Kayıtları")
c1, c2 = st.columns(2)
mod_filtre = c1.selectbox("Mod filtrele", ["Tümü", "D_tam", "A_ham"])
karar_filtre = c2.selectbox("Karar filtrele (Tarama eşiği)", ["Tümü", "Pozitif", "Negatif"])

goster = df.copy()
if mod_filtre != "Tümü":
    goster = goster[goster["mod"] == mod_filtre]
if karar_filtre == "Pozitif":
    goster = goster[goster["risk_skoru"] >= 0.50]
elif karar_filtre == "Negatif":
    goster = goster[goster["risk_skoru"] < 0.50]

st.dataframe(goster, use_container_width=True, height=400)

st.divider()

# ── CSV indir ──────────────────────────────────────────────────────────
st.subheader("Dışa Aktar")
csv = goster.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="📥 CSV olarak indir",
    data=csv,
    file_name=f"mds_tahminler_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
)

st.divider()

# ── Çıkış ──────────────────────────────────────────────────────────────
if st.button("Çıkış Yap"):
    st.session_state.admin_giris = False
    st.rerun()
