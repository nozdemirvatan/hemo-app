"""
MDS Hemogram Risk Hesaplayıcı
==============================
Tek mod: Tam Mod (D_tam) — hemogram + referans aralığı
ROC-AUC ~0.851

Çalıştırmak için:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime

st.set_page_config(page_title="MDS Hemogram Risk Hesaplayıcı", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)
# ── Sabitler ───────────────────────────────────────────────────────────
HAM = ['Monosit', 'Lökosit', 'Nötrofil', 'Lenfosit', 'Eritrosit',
       'Hemoglobin', 'Hematokrit', 'Ortalama Hücresel Hacim',
       'Eritrosit Dağılım Genişliği', 'Trombosit']

DISPLAY = {
    'Monosit': ('Monosit', 'x10³/µL'),
    'Lökosit': ('Lökosit', 'x10³/µL'),
    'Nötrofil': ('Nötrofil', 'x10³/µL'),
    'Lenfosit': ('Lenfosit', 'x10³/µL'),
    'Eritrosit': ('Eritrosit', 'x10⁶/µL'),
    'Hemoglobin': ('Hemoglobin', 'g/dL'),
    'Hematokrit': ('Hematokrit', '%'),
    'Ortalama Hücresel Hacim': ('MCV', 'fL'),
    'Eritrosit Dağılım Genişliği': ('RDW', '%'),
    'Trombosit': ('Trombosit', 'x10³/µL'),
}

DEFAULT_REF = {
    'Monosit': (0.2, 1.0),
    'Lökosit': (4.0, 10.0),
    'Nötrofil': (2.0, 7.0),
    'Lenfosit': (1.0, 3.0),
    'Eritrosit': (4.2, 5.9),
    'Hemoglobin': (12.0, 17.5),
    'Hematokrit': (36.0, 50.0),
    'Ortalama Hücresel Hacim': (80.0, 100.0),
    'Eritrosit Dağılım Genişliği': (11.5, 14.5),
    'Trombosit': (150.0, 400.0),
}

THR_TARAMA = 0.50
THR_DENGELI = 0.60


@st.cache_resource
def load_artifacts():
    return joblib.load("mds_artifacts.pkl")


def log_tahmin(mod, raw_values, prob):
    LOG_FILE = "tahmin_log.csv"
    kayit = {
        "tarih_saat": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mod": mod,
        "risk_skoru": round(prob, 4),
        "karar_tarama": "Pozitif" if prob >= THR_TARAMA else "Negatif",
        "karar_dengeli": "Pozitif" if prob >= THR_DENGELI else "Negatif",
    }
    kayit.update({k: v for k, v in raw_values.items()})
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=kayit.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(kayit)


def build_dtam_vector(raw_values, ref_ranges, feature_cols):
    row = {}
    for feat in HAM:
        v = raw_values.get(feat)
        row[feat] = v if v is not None else np.nan
        ref = ref_ranges.get(feat)
        rd_col = f"{feat} Referans Durum"
        norm_col = f"{feat} Normalize"
        miss_col = f"{feat} Normalize_Missing"
        if v is None or ref is None:
            row[rd_col] = 0.0
            row[norm_col] = np.nan
            row[miss_col] = 1
        else:
            alt, ust = ref
            if v < alt:
                row[rd_col] = -1.0
            elif v > ust:
                row[rd_col] = 1.0
            else:
                row[rd_col] = 0.0
            row[norm_col] = (v - alt) / (ust - alt) if ust != alt else np.nan
            row[miss_col] = 0
    return np.array([[row[c] for c in feature_cols]])


def render_decision(prob):
    c1, c2 = st.columns(2)
    with c1:
        label = "🔴 Pozitif — ileri tetkik önerilir" if prob >= THR_TARAMA else "🟢 Negatif"
        st.metric("Tarama Senaryosu (eşik = 0.50)", f"{prob:.3f}")
        st.caption(label)
    with c2:
        label = "🔴 Pozitif — ileri tetkik önerilir" if prob >= THR_DENGELI else "🟢 Negatif"
        st.metric("Dengeli Senaryo (Youden eşik ≈ 0.60)", f"{prob:.3f}")
        st.caption(label)


def render_shap(explainer, X_imputed, feature_cols):
    shap_values = explainer.shap_values(X_imputed)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        base_value = explainer.expected_value[1]
    elif shap_values.ndim == 3:
        sv = shap_values[0, :, 1]
        base_value = explainer.expected_value[1]
    else:
        sv = shap_values[0]
        base_value = explainer.expected_value

    display_names = [
        c.replace(" Referans Durum", " (RD)")
         .replace(" Normalize_Missing", " (Eksik)")
         .replace(" Normalize", " (Norm)")
        for c in feature_cols
    ]

    explanation = shap.Explanation(
        values=sv,
        base_values=base_value,
        data=X_imputed[0],
        feature_names=display_names,
    )

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(display_names))))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Arayüz ─────────────────────────────────────────────────────────────
st.title("🩸 MDS Hemogram Risk Hesaplayıcı")
st.caption(
    "Miyelodisplastik Sendrom (MDS) için rutin hemogram parametrelerine dayalı "
    "risk skorlama aracı. Bu araç klinik karar destek amaçlıdır, tek başına "
    "tanı koymak için kullanılmamalıdır."
)

artifacts = load_artifacts()
art = artifacts["D_tam"]

st.divider()
st.subheader("Hemogram Değerleri ve Referans Aralıkları")
st.caption("Referans alt/üst sınırları laboratuvarınıza göre düzenleyebilirsiniz.")

raw_values = {}
ref_ranges = {}

cols = st.columns(2)
for i, feat in enumerate(HAM):
    label, unit = DISPLAY[feat]
    default_alt, default_ust = DEFAULT_REF[feat]
    with cols[i % 2]:
        st.markdown(f"**{label}** ({unit})")
        c_val, c_alt, c_ust = st.columns([1.3, 1, 1])
        val = c_val.number_input(
            f"Değer##{feat}", min_value=0.0, value=None,
            step=0.1, key=f"val_{feat}",
            label_visibility="collapsed", placeholder="Değer"
        )
        alt = c_alt.number_input(
            f"Alt##{feat}", value=float(default_alt),
            step=0.1, key=f"alt_{feat}",
            label_visibility="collapsed"
        )
        ust = c_ust.number_input(
            f"Üst##{feat}", value=float(default_ust),
            step=0.1, key=f"ust_{feat}",
            label_visibility="collapsed"
        )
        raw_values[feat] = val
        ref_ranges[feat] = (alt, ust)

st.divider()

if st.button("Risk Hesapla", type="primary", use_container_width=True):
    X = build_dtam_vector(raw_values, ref_ranges, art["feature_cols"])
    X_imputed = art["imputer"].transform(X)
    prob = art["model"].predict_proba(X_imputed)[0, 1]
    log_tahmin("D_tam", raw_values, prob)

    st.subheader("Sonuç")
    st.info(f"Model: **D_tam** — ROC-AUC {art['oof_auc']:.3f}")
    render_decision(prob)

    st.subheader("SHAP Açıklaması — Riski Etkileyen Parametreler")
    render_shap(art["explainer"], X_imputed, art["feature_cols"])

st.divider()
st.caption(
    "⚠️ Bu araç araştırma/karar-destek amaçlıdır. Klinik kararlar "
    "hekim değerlendirmesiyle birlikte alınmalıdır."
)
