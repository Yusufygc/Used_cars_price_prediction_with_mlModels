# ================================================
# 🚗 Araç Fiyat Tahmini — Model Karşılaştırma (SVM, RandomForest, XGBoost)
# ================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------
# 1. Veri Yükleme
# ------------------------------------------------
DATA_PATH = "ml/GPT5_cleaned_and_engineered_cars.csv"

print("\n--- Adım 1: Veri Yükleniyor ---")
df = pd.read_csv(DATA_PATH)
print(f"Veri başarıyla yüklendi. Satır sayısı: {len(df)}, Sütun sayısı: {len(df.columns)}")

# ------------------------------------------------
# 2. Hedef ve Özellik Ayrımı
# ------------------------------------------------
if 'price' not in df.columns:
    raise ValueError("❌ 'price' kolonu veri setinde bulunamadı!")

X = df.drop(columns=['price'])
y = df['price']

# Fiyat aralığı geniş olduğu için log dönüşümü
y_log = np.log1p(y)

# ------------------------------------------------
# 3. Eğitim-Test Ayrımı
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"Eğitim kümesi: {X_train.shape}, Test kümesi: {X_test.shape}")

# ------------------------------------------------
# 4. Ölçekleme (SVM için önemli)
# ------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------
# 5. Model Tanımları
# ------------------------------------------------
models = {
    "SVM (RBF Kernel)": SVR(kernel='rbf', C=10, epsilon=0.2),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
}

# ------------------------------------------------
# 6. Eğitim ve Değerlendirme
# ------------------------------------------------
results = []

for name, model in models.items():
    print(f"\n--- {name} Eğitiliyor ---")
    if name.startswith("SVM"):
        model.fit(X_train_scaled, y_train)
        y_pred_log = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_log = model.predict(X_test)
    
    # Log'dan geri dönüştür
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results.append((name, mae, r2))

    print(f"{name} -> MAE: {mae:,.2f}, R²: {r2:.4f}")

# ------------------------------------------------
# 7. Sonuçların Tablosu
# ------------------------------------------------
results_df = pd.DataFrame(results, columns=["Model", "MAE", "R2_Score"]).sort_values(by="R2_Score", ascending=False)
print("\n\n--- 📊 Model Karşılaştırma Sonuçları ---")
print(results_df.to_string(index=False))

# ------------------------------------------------
# 8. En İyi Modelin Özeti
# ------------------------------------------------
best_model = results_df.iloc[0]
print(f"\n🏆 En İyi Model: {best_model['Model']} | MAE: {best_model['MAE']:,.2f} | R²: {best_model['R2_Score']:.4f}")

print("\n✅ Karşılaştırma tamamlandı.")
