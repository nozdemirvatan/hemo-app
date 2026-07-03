"""
Final production modelleri eğit ve kaydet (A_ham ve D_tam).
Fold yok - tüm veri üzerinde, canlı kullanım için tek model.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import shap

df = pd.read_csv('MDS_modelleme_v2.csv')

HAM = ['Monosit','Lökosit','Nötrofil','Lenfosit','Eritrosit',
       'Hemoglobin','Hematokrit','Ortalama Hücresel Hacim',
       'Eritrosit Dağılım Genişliği','Trombosit']
REF_DURUM = [c for c in df.columns if 'Referans Durum' in c]
NORMALIZE = [c for c in df.columns if 'Normalize' in c and '_Missing' not in c]
MISSING_IND = [c for c in df.columns if '_Missing' in c]

FEATURE_SETS = {
    "A_ham": HAM,
    "D_tam": HAM + REF_DURUM + NORMALIZE + MISSING_IND,
}

y = df['mds'].values
groups = df['Hasta_ID'].values

def compute_sample_weight(y_arr, groups_arr):
    id_counts = pd.Series(groups_arr).value_counts()
    pw = np.array([1/id_counts[g] for g in groups_arr])
    n_total = len(y_arr)
    n_neg = (y_arr == 0).sum()
    n_pos = (y_arr == 1).sum()
    cw = {0: n_total/(2*n_neg), 1: n_total/(2*n_pos)}
    sw = pw * np.array([cw[yi] for yi in y_arr])
    return sw

artifacts = {}

# Quick out-of-fold sanity check (not the final model, just to report expected AUC)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for set_name, feat_cols in FEATURE_SETS.items():
    X_full = df[feat_cols].values
    oof_prob = np.zeros(len(y))
    for train_idx, test_idx in sgkf.split(X_full, y, groups):
        imputer_cv = SimpleImputer(strategy='median')
        X_tr = imputer_cv.fit_transform(X_full[train_idx])
        X_te = imputer_cv.transform(X_full[test_idx])
        sw = compute_sample_weight(y[train_idx], groups[train_idx])
        m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        m.fit(X_tr, y[train_idx], sample_weight=sw)
        oof_prob[test_idx] = m.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y, oof_prob)
    print(f"{set_name}: OOF ROC-AUC = {auc:.3f}")

    # ── Final model: tüm veri üzerinde eğit (production) ──
    imputer_final = SimpleImputer(strategy='median')
    X_final = imputer_final.fit_transform(X_full)
    sw_final = compute_sample_weight(y, groups)
    model_final = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model_final.fit(X_final, y, sample_weight=sw_final)

    explainer = shap.TreeExplainer(model_final)

    artifacts[set_name] = {
        "model": model_final,
        "imputer": imputer_final,
        "feature_cols": feat_cols,
        "explainer": explainer,
        "oof_auc": auc,
    }

joblib.dump(artifacts, 'mds_artifacts.pkl')
print("\nKaydedildi: mds_artifacts.pkl")
print("Feature listeleri:")
for k, v in artifacts.items():
    print(f"  {k}: {len(v['feature_cols'])} feature")
