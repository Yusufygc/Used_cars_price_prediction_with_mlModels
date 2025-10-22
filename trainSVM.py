# ================================================
# 🚗 Araç Fiyat Tahmini — SVM (Support Vector Regression)
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

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

print(f"Bağımsız değişken sayısı: {X.shape[1]}")
print(f"Bağımlı değişken (price) örnek ortalaması: {y.mean():,.2f}")

# ------------------------------------------------
# 3. Hedef Değişken Dönüşümü (log1p)
# ------------------------------------------------
y_log = np.log1p(y)

# ------------------------------------------------
# 4. Eğitim-Test Ayrımı
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"Eğitim kümesi: {X_train.shape}, Test kümesi: {X_test.shape}")

# ------------------------------------------------
# 5. Özellik Ölçekleme
# ------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Veri ölçekleme tamamlandı (StandardScaler).")

# ------------------------------------------------
# 6. Model Tanımı ve Eğitim
# ------------------------------------------------
print("\n--- Adım 6: SVM Model Eğitimi Başladı ---")

# İlk hızlı test modeli
svr = SVR(kernel='rbf', C=10, epsilon=0.2)
svr.fit(X_train_scaled, y_train)

# Tahminler (log veriden geri dönüştürülmüş)
y_pred_log = svr.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# ------------------------------------------------
# 7. Performans Metrikleri
# ------------------------------------------------
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n--- Model Performansı ---")
print(f"Ortalama Mutlak Hata (MAE): {mae:,.2f}")
print(f"R² Skoru: {r2:.4f}")

# ------------------------------------------------
# 8. Hata Dağılım Grafiği
# ------------------------------------------------
errors = y_true - y_pred
plt.figure(figsize=(8,5))
plt.hist(errors, bins=50, color='steelblue', edgecolor='black')
plt.title("Tahmin Hatalarının Dağılımı")
plt.xlabel("Gerçek Fiyat - Tahmin Fiyatı")
plt.ylabel("Frekans")
plt.show()

# ------------------------------------------------
# 9. Gerçek vs Tahmin Grafiği
# ------------------------------------------------
plt.figure(figsize=(7,7))
plt.scatter(y_true, y_pred, alpha=0.5, color='orange', edgecolor='black')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
plt.title("Gerçek vs Tahmin Fiyatları (SVM)")
plt.xlabel("Gerçek Fiyat ($)")
plt.ylabel("Tahmin Fiyat ($)")
plt.show()

# ------------------------------------------------
# 10. Grid Search (Opsiyonel: En iyi hiperparametreleri bul)
# ------------------------------------------------
# Yorum satırından kaldırarak çalıştırabilirsin:
"""
param_grid = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'C': [1, 10, 50],
    'epsilon': [0.1, 0.2, 0.5]
}
grid = GridSearchCV(SVR(), param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
grid.fit(X_train_scaled, y_train)
print("En iyi parametreler:", grid.best_params_)
print("En iyi R² skoru:", grid.best_score_)
"""

print("\n✅ Eğitim tamamlandı. Sonuçlar yukarıda gösterildi.")
