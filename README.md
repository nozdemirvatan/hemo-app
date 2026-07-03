# MDS Hemogram Risk Hesaplayıcı

## Dosyalar
- `app.py` — Streamlit uygulaması (iki modlu: Tam / Hızlı)
- `mds_artifacts.pkl` — Eğitilmiş modeller, imputer'lar ve SHAP explainer'lar (D_tam + A_ham)
- `train_models.py` — Modelleri sıfırdan yeniden eğitmek istersen (MDS_modelleme_v2.csv gerekir)
- `requirements.txt` — Bağımlılıklar

## Kurulum
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma
```bash
streamlit run app.py
```
Tarayıcıda otomatik olarak `http://localhost:8501` açılır.

## Modeller hakkında
| Mod | Feature seti | OOF ROC-AUC | Girdi |
|---|---|---|---|
| Tam Mod | D_tam (40 özellik) | 0.851 | 10 hemogram değeri + her biri için referans alt/üst sınır |
| Hızlı Mod | A_ham (10 özellik) | 0.826 | Sadece 10 hemogram değeri |

Her iki model de `StratifiedGroupKFold` (5 fold, hasta bazlı gruplama) ile
`mds_modelleme_sablonu.py`'deki ağırlıklandırma mantığı (hasta ağırlığı × sınıf
ağırlığı) kullanılarak eğitildi. Üretim modelleri tüm veri seti üzerinde
(531 hasta / 1.888 kayıt) yeniden eğitilmiştir; raporlanan AUC değerleri
out-of-fold tahminlerden hesaplanmıştır.

## Referans Durum / Normalize hesaplama
- **Referans Durum**: değer < alt sınır → −1, alt ≤ değer ≤ üst → 0, değer > üst sınır → +1
- **Normalize**: (değer − alt sınır) / (üst sınır − alt sınır)
- Referans bilgisi girilmezse: Normalize = NaN (eğitim medyanıyla impute edilir),
  Referans Durum = 0 (normal kabul edilir), ilgili `_Missing` göstergesi = 1

## Karar eşikleri
- **Tarama senaryosu**: eşik = 0.50 (yüksek duyarlılık, daha fazla pozitif çağırır)
- **Dengeli senaryo**: eşik ≈ 0.60 (Youden J istatistiğinden türetilmiş)

## Varsayılan referans aralıkları
Tam Mod'da her parametre için genel yetişkin referans aralığı önceden
doldurulmuştur (ör. Hemoglobin 12.0–17.5 g/dL). Bunlar **yaklaşık** değerlerdir;
kullanmadan önce kendi laboratuvarınızın referans aralıklarıyla güncelleyin.

## Önemli not
Bu araç klinik karar destek amaçlıdır, tek başına tanı koymak için
kullanılmamalıdır. Sonuçlar hekim değerlendirmesiyle birlikte yorumlanmalıdır.
