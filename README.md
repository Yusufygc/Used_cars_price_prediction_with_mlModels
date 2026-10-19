# İkinci El Araç Fiyat Tahmini

Ham `data/used_cars.csv` dosyasından fiyat tahmini yapan, leakage içermeyen regresyon pipeline'ı.
İlk sürümün hataları için bkz. [`LESSONS.md`](LESSONS.md). Eski notebook'lar `legacy/` altında, referans içindir.

## Kurulum

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Çalıştırma

```powershell
.\.venv\Scripts\python -m pytest            # ayrıştırıcı testleri
.\.venv\Scripts\python -m src.train         # split, CV ile seçim, test (tek sefer), models/best_pipeline.joblib
.\.venv\Scripts\python -m src.explain       # SHAP grafikleri, reports/
.\.venv\Scripts\python -m src.predict --json '{...ham kayıt...}'
```

## Tasarım kararları

- **Tek girdi**: `data/used_cars.csv`. Ara CSV yok.
- **Split ilk adım** (%80/%20). Test seti yalnızca seçilen model için bir kez kullanılır.
- **Model seçimi**: train üzerinde `RandomizedSearchCV` + tekrarlı 5-fold CV (MAE). Adaylar: Ridge, SVR, Random Forest, XGBoost. Baseline: medyan.
- **Öğrenen tüm adımlar Pipeline içinde**: medyan doldurma (+eksiklik göstergesi), ölçekleme, one-hot, `TargetEncoder` (brand, model).
- **Hedef**: `log1p(price)`, tahmin dolar döner (`TransformedTargetRegressor`).
- **NaN anlamlı kategori**: `clean_title=missing`, `accident=unknown`.
- **Satır silinmez** (yalnızca hedef, yaş veya mil eksikse). Aşırı pahalı araçlar veride kalır, log hedef etkilerini bastırır.
- **PCA ve filtre tabanlı feature selection kullanılmaz**: XAI için orijinal özellik adları korunur.
- **Tahmin aralığı**: CV artıklarının log-uzayında %90 kantili.
- **Birim**: kilometre değil, **mil**.
- **Kapsam**: veri ABD ilan verisidir (tek kaynak, ~4000 satır). Çok nadir markalar için tahmin güvenilir değildir.

## Çıktılar (`reports/`, `models/` git'e girmez)

`cv_results.csv`, `metrics.json`, `error_analysis.csv`, `test_predictions.csv`, `shap_*.png`, `shap_importance.csv`.
