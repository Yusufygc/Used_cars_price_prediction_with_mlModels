#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-28T16:22:28.460Z
"""

# Gerekli Kütüphanelerin Yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings


# Uyarıları kapatmak için (Daha temiz bir çıktı için)
warnings.filterwarnings('ignore')

# Görselleştirme ayarları (Profesyonel görünüm için)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("Kütüphaneler başarıyla yüklendi.")

# 1. VERİ YÜKLEME
# ==========================================
try:
    df = pd.read_csv('data/used_cars.csv')
    print("Veri seti başarıyla yüklendi!")
except FileNotFoundError:
    print("Hata: Dosya bulunamadı. Lütfen dosya yolunu kontrol edin.")

# 2. VERİYE İLK BAKIŞ (EDA - Initial Exploration)
# ==========================================

# İlk 5 satırı görüntüleme
print("\n--- Veri Setinin İlk 5 Satırı ---")
display(df.head())

# Veri Setinin Boyutu
print(f"\n--- Veri Seti Boyutu ---")
print(f"Satır Sayısı: {df.shape[0]}")
print(f"Sütun Sayısı: {df.shape[1]}")

# Veri Tipleri ve Bellek Kullanımı
print("\n--- Veri Seti Hakkında Genel Bilgi ---")
df.info()

# Tekrar Eden Verilerin Kontrolü (Duplicates)
duplicate_count = df.duplicated().sum()
print(f"\n--- Tekrar Eden Veri Sayısı ---")
print(f"Toplam Duplicate Satır: {duplicate_count}")

# 3. EKSİK VERİLERİN (MISSING VALUES) ÖZETİ
# ==========================================
print("\n--- Sütun Bazlı Eksik Veri Sayıları ---")
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percentage = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

missing_df = pd.concat([missing_values, missing_percentage], axis=1, keys=['Toplam Eksik', 'Yüzde (%)'])
print(missing_df[missing_df['Toplam Eksik'] > 0]) # Sadece eksik verisi olanları göster

# Eksik Veri Görselleştirmesi (Heatmap)
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Veri Setindeki Eksik Verilerin Dağılımı (Sarı Çizgiler Eksik Veriyi Temsil Eder)')
plt.show()

# ### Temizleme Aşaması


print("--- Temizleme Öncesi Veri Tipleri ---")
print(df[['price', 'milage']].dtypes)
print("\n" + "="*50 + "\n")

# 1. PRICE (FİYAT) SÜTUNUNU TEMİZLEME
# ========================================================
# '$' ve ',' işaretlerini kaldırıp sayıya çeviriyoruz.
# Hata almamak için önce stringe çevirip işlem yapıyoruz.

try:
    df['price'] = df['price'].astype(str).str.replace('$', '', regex=False)
    df['price'] = df['price'].str.replace(',', '', regex=False)
    df['price'] = pd.to_numeric(df['price'], errors='coerce') # Dönüştürülemeyenleri NaN yapar
    
    print(" 'Price' sütunu başarıyla sayısal veriye dönüştürüldü.")
except Exception as e:
    print(f"Price dönüştürme hatası: {e}")

# 2. MILAGE (KİLOMETRE) SÜTUNUNU TEMİZLEME
# ========================================================
# 'mi.' ve ',' işaretlerini kaldırıyoruz.

try:
    df['milage'] = df['milage'].astype(str).str.replace('mi.', '', regex=False)
    df['milage'] = df['milage'].str.replace(',', '', regex=False)
    df['milage'] = pd.to_numeric(df['milage'], errors='coerce')
    
    print(" 'Milage' sütunu başarıyla sayısal veriye dönüştürüldü.")
except Exception as e:
    print(f"Milage dönüştürme hatası: {e}")


print("\n" + "="*50 + "\n")
print("--- Temizleme Sonrası İstatistiksel Özet ---")
print(df[['price', 'milage']].describe())

# Grafik Alanı Oluşturma (2 Satır, 2 Sütun)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Fiyat Dağılımı (Histogram)
sns.histplot(df['price'], bins=50, kde=True, color='green', ax=axes[0, 0])
axes[0, 0].set_title('Fiyat Dağılımı (Histogram)')
axes[0, 0].set_xlabel('Fiyat ($)')

# Fiyat Aykırı Değer Analizi (Boxplot)
sns.boxplot(x=df['price'], color='lightgreen', ax=axes[0, 1])
axes[0, 1].set_title('Fiyat Aykırı Değer Analizi (Boxplot)')

# Kilometre Dağılımı (Histogram)
sns.histplot(df['milage'], bins=50, kde=True, color='blue', ax=axes[1, 0])
axes[1, 0].set_title('Kilometre Dağılımı (Histogram)')
axes[1, 0].set_xlabel('Mil (Mile)')

# Kilometre Aykırı Değer Analizi (Boxplot)
sns.boxplot(x=df['milage'], color='lightblue', ax=axes[1, 1])
axes[1, 1].set_title('Kilometre Aykırı Değer Analizi (Boxplot)')

plt.tight_layout()
plt.show()

# Fiyat ve Kilometre Arasındaki İlişki (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='milage', y='price', data=df, alpha=0.5)
plt.title('Fiyat vs. Kilometre İlişkisi')
plt.xlabel('Kilometre (Mil)')
plt.ylabel('Fiyat ($)')
plt.show()

# ## FUEL


print("--- Ham Fuel_Type (Yakıt) Verisi ---")
print(f"Benzersiz Değer Sayısı: {df['fuel_type'].nunique()}")
print(f"Eksik Veri Sayısı: {df['fuel_type'].isnull().sum()}")
print("\nEn Sık Görülen 10 Değer:")
print(df['fuel_type'].value_counts().head(10))
print("\n" + "="*50 + "\n")


print(f"Toplam Eksik (NaN): {df['fuel_type'].isnull().sum()}")
print("Örnek Eksik Veriler (Markalar):")
print(df[df['fuel_type'].isnull()]['brand'].unique())
print("\n" + "="*50 + "\n")

# 2. AKILLI YAKIT SINIFLANDIRMA (SMART CLASSIFICATION)
# ========================================================

# Sadece elektrikli araç üreten markalar listesi
# Bu markaları görürsek direkt 'Electric' etiketi basacağız.
EV_BRANDS = ['Tesla', 'Rivian', 'Lucid', 'Polestar', 'Fisker', 'Karma']

def classify_fuel_smart(row):
    # Verileri küçük harfe çevirip string yapalım
    brand = str(row['brand'])
    fuel = str(row['fuel_type']).lower()
    engine = str(row['engine']).lower()
    model = str(row['model']).lower()
    
    # KURAL 1: MARKA BAZLI KESİN ATAMA
    # Eğer marka Tesla, Rivian vb. ise tartışmasız Elektriktir.
    if brand in EV_BRANDS:
        return 'Electric'
    
    # KURAL 2: HİBRİT KONTROLÜ (Öncelikli)
    # Hem elektrik hem benzin kelimesi geçiyorsa Hibrittir.
    if ('hybrid' in fuel) or ('hybrid' in engine) or ('plug-in' in fuel):
        return 'Hybrid'
    
    # KURAL 3: ELEKTRİK KELİME KONTROLÜ
    # Marka listede yok ama motor bilgisinde 'Electric' yazıyor olabilir.
    if 'electric' in fuel or 'electric' in engine or 'battery' in engine:
        return 'Electric'
        
    # KURAL 4: DİĞER YAKITLAR
    if 'diesel' in fuel:
        return 'Diesel'
    if 'e85' in fuel or 'flex' in fuel:
        return 'E85 Flex Fuel'
    if 'gas' in fuel or 'petrol' in fuel:
        return 'Gasoline'
    
    # KURAL 5: HALA BULAMADIKLARIMIZ
    # Eğer fuel_type NaN ise ve yukarıdakilere girmediyse
    if pd.isna(row['fuel_type']) or row['fuel_type'] == 'nan':
        return 'Unknown'
        
    return 'Gasoline' # Varsayılan olarak en yaygın yakıtı atayabiliriz veya 'Other' diyebiliriz.

# Fonksiyonu satır satır (axis=1) uyguluyoruz
df['fuel_category'] = df.apply(classify_fuel_smart, axis=1)

# Unknown kalanları en çok tekrar eden (Mod) ile doldurma (Opsiyonel, şu an analiz için bırakabiliriz)
# df['fuel_category'] = df['fuel_category'].replace('Unknown', df['fuel_category'].mode()[0])

print("--- Akıllı Sınıflandırma Sonrası Dağılım ---")
print(df['fuel_category'].value_counts())

# Kontrol: Tesla araçları ne olarak etiketlendi?
print("\n--- Tesla Marka Araçların Yakıt Etiketleri ---")
print(df[df['brand'] == 'Tesla'][['brand', 'model', 'fuel_type', 'fuel_category']].head())

print("\n" + "="*50 + "\n")

# 3. GÖRSELLEŞTİRME
# ========================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Adet Dağılımı
sns.countplot(x='fuel_category', data=df, palette='coolwarm', order=df['fuel_category'].value_counts().index, ax=axes[0])
axes[0].set_title('Yakıt Tipi Dağılımı (Akıllı Doldurma Sonrası)')
axes[0].bar_label(axes[0].containers[0])

# Fiyat Dağılımı
sns.boxplot(x='fuel_category', y='price', data=df, palette='Set2', ax=axes[1])
axes[1].set_title('Yakıt Tipine Göre Fiyat Dağılımı')
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()

# 4. İSTATİSTİKSEL ÖZET
# ========================================================
summary = df.groupby('fuel_category')[['price', 'milage']].agg(['mean', 'median', 'count', 'min', 'max'])
display(summary.round(0))

display(df.head())

# ## Transmission


# ========================================================
# BÖLÜM 1: KAOSU GÖSTER (HAM VERİ ANALİZİ)
# ========================================================
print("--- Ham Vites Verisi İncelemesi ---")
print(f"Toplam Farklı Vites Tanımı Sayısı: {df['transmission'].nunique()}")
print("Örnek Karmaşık Veriler:")
print(df['transmission'].unique()[:10])

# Görselleştirme: En sık geçen ilk 20 karmaşık ifade
top_20_raw = df['transmission'].value_counts().head(20)

plt.figure(figsize=(14, 6))
sns.barplot(x=top_20_raw.values, y=top_20_raw.index, palette='rocket')
plt.title(f"Ham Verideki Karmaşa (Toplam {df['transmission'].nunique()} Farklı Tip Var!)")
plt.xlabel('Araç Sayısı')
plt.ylabel('Ham Vites Tanımı')
plt.show()

# ========================================================
# BÖLÜM 2: SİHİRLİ DOKUNUŞ (TEMİZLEME KURALLARI)
# ========================================================
# Burada sınıfa "Bu karmaşayı çözmek için akıllı bir algoritma yazdım" diyebilirsin.

EV_BRANDS = ['Tesla', 'Rivian', 'Lucid', 'Polestar', 'Fisker', 'Karma']

# Önce yakıtı belirle (Yardımcı fonksiyon)
def classify_fuel_helper(row):
    brand = str(row['brand'])
    fuel = str(row['fuel_type']).lower()
    engine = str(row['engine']).lower()
    if brand in EV_BRANDS: return 'Electric'
    if 'hybrid' in fuel or 'hybrid' in engine: return 'Hybrid'
    if 'electric' in fuel or 'battery' in engine: return 'Electric'
    return 'Gasoline'

df['fuel_category'] = df.apply(classify_fuel_helper, axis=1)

# Vites Temizleme Fonksiyonu
def clean_transmission_advanced(row):
    text = str(row['transmission']).upper()
    fuel = row['fuel_category']
    
    # Kural 1: Elektrikli -> Single-Speed
    if fuel == 'Electric':
        return 'Single-Speed'
    # Kural 2: Manuel
    if 'MANUAL' in text or 'M/T' in text or 'MT' in text:
        return 'Manual'
    # Kural 3: CVT
    if 'CVT' in text:
        return 'CVT'
    # Kural 4: Yarı Otomatik (DCT/Dual Shift)
    if 'DUAL SHIFT' in text or 'DCT' in text or 'SEMI' in text or 'PDK' in text or 'DSG' in text:
        return 'Semi-Auto'
    # Kural 5: Diğerleri Otomatik
    return 'Automatic'

# Fonksiyonu çalıştır
df['transmission_type'] = df.apply(clean_transmission_advanced, axis=1)


# ========================================================
# BÖLÜM 3: DÜZENİ GÖSTER (TEMİZ VERİ ANALİZİ)
# ========================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 1. Grafik: Temizlenmiş Dağılım
order = df['transmission_type'].value_counts().index
sns.countplot(x='transmission_type', data=df, palette='viridis', order=order, ax=axes[0])
axes[0].set_title('Temizleme Sonrası Kategoriler (Net ve Anlaşılır)')
axes[0].bar_label(axes[0].containers[0])

# 2. Grafik: Fiyat Etkisi
sns.boxplot(x='transmission_type', y='price', data=df, palette='coolwarm', ax=axes[1])
axes[1].set_title('Vites Tipine Göre Fiyat Dağılımı')
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()

# İstatistiksel Kanıt
print("\n--- Temizlik Sonrası Özet ---")
print(df['transmission_type'].value_counts())

display(df.groupby('transmission_type')[['price', 'milage']].agg(['mean', 'median', 'count', 'min', 'max']).round(0))

display(df.head())

# ### Engine


# --- KRİTİK ADIM: MOTOR ANALİZİ İÇİN YAKIT TÜRÜNÜ BİLMEK ŞART ---
# Önceki adımlarda belirlediğimiz akıllı yakıt sınıflandırmasını burada uyguluyoruz.
EV_BRANDS = ['Tesla', 'Rivian', 'Lucid', 'Polestar', 'Fisker', 'Karma']
def classify_fuel_helper(row):
    brand = str(row['brand'])
    fuel = str(row['fuel_type']).lower()
    engine = str(row['engine']).lower()
    if brand in EV_BRANDS: return 'Electric'
    if 'hybrid' in fuel or 'hybrid' in engine: return 'Hybrid'
    if 'electric' in fuel or 'battery' in engine: return 'Electric'
    if 'diesel' in fuel: return 'Diesel'
    return 'Gasoline'

df['fuel_category'] = df.apply(classify_fuel_helper, axis=1)

# ========================================================
# BÖLÜM 1: HAM VERİ ANALİZİ (KAOSU GÖSTER)
# ========================================================
print("--- 1. HAM ENGINE (MOTOR) VERİ ANALİZİ ---")
unique_engines = df['engine'].nunique()
missing_engines = df['engine'].isnull().sum()
print(f"Toplam Benzersiz Motor Metni Sayısı: {unique_engines}")
print(f"Eksik (Null) Motor Verisi: {missing_engines}")
print("\nEn Sık Karşılaşılan 5 Karmaşık Veri:")
print(df['engine'].value_counts().head(5))

# Görselleştirme: Ham verinin karmaşası
top_15_engine = df['engine'].value_counts().head(15)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_15_engine.values, y=top_15_engine.index, palette='Reds_r')
plt.title(f'Veri Setindeki En Yaygın 15 Motor Tipi (Toplam {unique_engines} Çeşit Var!)')
plt.xlabel('Araç Sayısı')
plt.show()

# ========================================================
# BÖLÜM 2: TEMİZLİK VE AYRIŞTIRMA (SİHİRLİ DOKUNUŞ)
# ========================================================
# Burada Regex ve Mantıksal Kurallar devreye giriyor.

def extract_engine_features(row):
    text = str(row['engine'])
    fuel = row['fuel_category']
    
    # Başlangıç değerleri
    hp, liters, cylinders = None, None, None
    
    # 1. Beygir Gücü (HP) - Her araçta aranır
    # Örn: "300.0HP" ifadesini yakalar
    hp_match = re.search(r'(\d+\.?\d*)HP', text)
    if hp_match:
        hp = float(hp_match.group(1))
        
    # 2. Motor Hacmi ve Silindir (Yakıta Göre Değişir)
    if fuel == 'Electric':
        # Elektrikli araçlarda litre ve silindir 0 kabul edilir.
        liters = 0.0
        cylinders = 0.0
    else:
        # İçten yanmalı araçlarda Litre ve Silindir aranır
        
        # Litre (Örn: 3.5L)
        liters_match = re.search(r'(\d+\.?\d*)\s*L', text)
        if liters_match:
            liters = float(liters_match.group(1))
            
        # Silindir (Örn: V6, 6 Cylinder)
        cyl_match = re.search(r'V(\d+)', text)
        if not cyl_match:
            cyl_match = re.search(r'(\d+)\s*Cylinder', text)
        
        if cyl_match:
            cylinders = int(cyl_match.group(1))
            
    return pd.Series([hp, liters, cylinders])

# Fonksiyonu uygulama
print("\n--- 2. ÖZELLİK ÇIKARIMI (FEATURE EXTRACTION) YAPILIYOR ---")
df[['horsepower', 'engine_volume', 'cylinders']] = df.apply(extract_engine_features, axis=1)
print("İşlem Tamamlandı. Metinler sayılara dönüştürüldü.")


# ========================================================
# BÖLÜM 3: SONUÇ ANALİZİ VE KARŞILAŞTIRMA
# ========================================================

# İstatistiksel Özet
print("\n--- 3. YENİ SÜTUNLARIN İSTATİSTİKLERİ ---")
print(df[['horsepower', 'engine_volume', 'cylinders']].describe().round(1))

# Görselleştirme: Temiz Verinin Dağılımı
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Beygir Gücü
sns.histplot(df['horsepower'], bins=30, kde=True, color='red', ax=axes[0])
axes[0].set_title('Beygir Gücü (HP) Dağılımı')

# Motor Hacmi
sns.histplot(df['engine_volume'], bins=20, kde=True, color='blue', ax=axes[1])
axes[1].set_title('Motor Hacmi (Litre) Dağılımı')
axes[1].set_xlabel('Litre (Elektrikli Araçlar 0.0)')

# Silindir Sayısı
sns.countplot(x='cylinders', data=df, palette='viridis', ax=axes[2])
axes[2].set_title('Silindir Sayısı Dağılımı')
axes[2].set_xlabel('Silindir (Elektrikli Araçlar 0)')

plt.tight_layout()
plt.show()

# ========================================================
# BÖLÜM 4: EKSİK VERİ KONTROLÜ (YENİ DEĞERLER)
# ========================================================
print("\n--- 4. YENİ OLUŞAN SÜTUNLARDAKİ EKSİK VERİ SAYILARI ---")
null_counts = df[['horsepower', 'engine_volume', 'cylinders']].isnull().sum()
print(null_counts)

print("\nNOT: Bu eksik veriler, metin içinde 'HP', 'L' veya 'Cylinder' yazmayan satırlardır.")
print("Bir sonraki adımda bunları akıllıca dolduracağız (Imputation).")

# Bu aşama sunumun için çok kritik bir "Kanıt" noktasıdır. Sınıfa şunu diyeceksin: "Eksik verileri rastgele doldurmadım. Önce o aracın Modeline baktım, yoksa Markasına baktım. Böylece verinin doğal yapısını bozmadım."


# ========================================================
# 2. AKILLI DOLDURMA (IMPUTATION) - KRİTİK ADIM
# ========================================================
print("--- Doldurma İşlemi Başlıyor ---")
cols_to_fill = ['horsepower', 'engine_volume', 'cylinders']

# Karşılaştırma yapabilmek için orijinal veriyi kopyalayalım
df_original = df[cols_to_fill].copy()

print(f"Başlangıçtaki Eksik Veriler:\n{df[cols_to_fill].isnull().sum()}")

# DÖNGÜ: Her sütun için aşamalı doldurma
for col in cols_to_fill:
    # Adım 1: Aynı MODELdeki araçların medyanını al (En hassas tahmin)
    # Örn: 'Ford F-150' eksikse, diğer 'Ford F-150'lerin ortalamasını bas.
    df[col] = df[col].fillna(df.groupby('model')[col].transform('median'))
    
    # Adım 2: Hala boşsa, aynı MARKAdaki araçların medyanını al
    # Örn: Modelden bulamadık, 'Ford' markasının genel ortalamasını bas.
    df[col] = df[col].fillna(df.groupby('brand')[col].transform('median'))
    
    # Adım 3: Çok nadir durumlarda hala boşsa, genel medyanı bas
    df[col] = df[col].fillna(df[col].median())

print(f"\nİşlem Sonrası Eksik Veriler:\n{df[cols_to_fill].isnull().sum()}")

# ========================================================
# 3. KANITLAMA (GÖRSEL KARŞILAŞTIRMA)
# ========================================================
# Burada amacımız, doldurma işleminin verinin dağılımını bozmadığını ispatlamak.

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_titles = [('horsepower', 'Beygir Gücü'), ('engine_volume', 'Motor Hacmi'), ('cylinders', 'Silindir')]

for i, (col, title) in enumerate(plot_titles):
    # Orijinal Veri (Kırmızı Çizgili) - Sadece mevcut veriler
    sns.kdeplot(df_original[col].dropna(), ax=axes[i], color='red', linestyle='--', label='Orijinal (Eksikler Hariç)')
    
    # Doldurulmuş Veri (Yeşil Dolgulu)
    sns.kdeplot(df[col], ax=axes[i], color='green', fill=True, alpha=0.3, label='Doldurulmuş (Final)')
    
    axes[i].set_title(f'{title} Dağılım Kontrolü')
    axes[i].legend()

plt.tight_layout()
plt.show()

# İstatistiksel Sağlama
print("\n--- İstatistiksel Kontrol (Sapma Var mı?) ---")
comparison = pd.DataFrame({
    'Orijinal Ort': df_original.mean(),
    'Yeni Ort': df[cols_to_fill].mean(),
    'Fark': df[cols_to_fill].mean() - df_original.mean()
})
print(comparison.round(3))

# ### accident ve clean_title


# ========================================================
# BÖLÜM 1: HAM VERİ ANALİZİ
# ========================================================
print("--- 1. Accident (Kaza) Sütunu ---")
print(df['accident'].value_counts(dropna=False))

print("\n--- 2. Clean Title (Temiz Belge) Sütunu ---")
print(df['clean_title'].value_counts(dropna=False))
print(f"Clean Title Eksik Veri Sayısı: {df['clean_title'].isnull().sum()}")

# ========================================================
# BÖLÜM 2: DÖNÜŞÜM VE ENCODING (0-1 DÖNÜŞÜMÜ)
# ========================================================

# 1. Accident Dönüşümü
# "None reported" -> 0 (Kaza Yok)
# "At least 1 accident..." -> 1 (Kaza Var)
# Eksik veri varsa (NaN), riskli olduğu için 1 kabul edilebilir veya moda göre doldurulabilir.
# Bu veri setinde genelde NaN yoktur ama kontrol edelim.

def clean_accident(text):
    if pd.isna(text): return 0 # Bilinmiyorsa temiz varsayalım (veya analize göre değişir)
    if 'None reported' in str(text):
        return 0
    else:
        return 1 # Kaza var

df['accident_binary'] = df['accident'].apply(clean_accident)

# 2. Clean Title Dönüşümü
# "Yes" -> 1 (Belge Temiz)
# NaN -> 0 (Belge Temiz Değil / Bilinmiyor / Riskli)
# Not: İlanlarda genelde "Clean Title" ise işaretlenir, değilse boş bırakılır.
# Bu yüzden boşları 0 (Riskli) kabul etmek mantıklı bir yaklaşımdır.

df['clean_title_binary'] = df['clean_title'].apply(lambda x: 1 if str(x) == 'Yes' else 0)
# ========================================================
print("\n--- Dönüşüm Sonrası Özet ---")
print("Kaza Durumu (1: Var, 0: Yok):")
print(df['accident_binary'].value_counts())
print("\nTemiz Belge (1: Evet, 0: Hayır/Bilinmiyor):")
print(df['clean_title_binary'].value_counts())

# ========================================================
# BÖLÜM 3: GÖRSELLEŞTİRME VE FİYAT ETKİSİ
# ========================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Grafik 1: Kaza Durumunun Fiyata Etkisi
sns.boxplot(x='accident_binary', y='price', data=df, palette='Reds', ax=axes[0])
axes[0].set_title('Kaza Durumunun Fiyata Etkisi')
axes[0].set_xticklabels(['Kazasız (0)', 'Kazalı (1)'])
axes[0].set_yscale('log')

# Grafik 2: Clean Title Durumunun Fiyata Etkisi
sns.boxplot(x='clean_title_binary', y='price', data=df, palette='Greens', ax=axes[1])
axes[1].set_title('Clean Title (Temiz Belge) Durumunun Fiyata Etkisi')
axes[1].set_xticklabels(['Belirsiz/Yok (0)', 'Temiz Belge (1)'])
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()

# İstatistiksel Kanıt (Ortalama Fiyat Farkları)
print("\n--- Fiyat Analizi ---")
acc_price = df.groupby('accident_binary')['price'].median()
title_price = df.groupby('clean_title_binary')['price'].median()

print(f"Kazasız Araç Medyan Fiyatı: ${acc_price[0]:,.0f}")
print(f"Kazalı Araç Medyan Fiyatı:  ${acc_price[1]:,.0f}")
print(f"--> Kaza, fiyatı ortalama %{((acc_price[0]-acc_price[1])/acc_price[0]*100):.1f} düşürüyor.\n")

print(f"Clean Title Olan Araç Medyan Fiyatı:   ${title_price[1]:,.0f}")
print(f"Clean Title Olmayan Araç Medyan Fiyatı: ${title_price[0]:,.0f}")
print(f"--> Clean Title, fiyatı ortalama %{((title_price[1]-title_price[0])/title_price[1]*100):.1f} artırıyor.")

# Clean Title Analizi
# 1 = Yes, 0 = NaN (Bizim varsayımımız buydu)
df['clean_title_binary'] = df['clean_title'].apply(lambda x: 1 if str(x) == 'Yes' else 0)

print("--- Clean Title PARADOKS ANALİZİ ---")

# Grup 0 (Bizim 'Belgesiz' dediklerimiz) ve Grup 1 (Belgeli) karşılaştırması
grup_0 = df[df['clean_title_binary'] == 0]
grup_1 = df[df['clean_title_binary'] == 1]

print(f"Grup 0 (NaN/Yok) Ort. Fiyat: ${grup_0['price'].mean():,.0f}")
print(f"Grup 1 (Yes/Var) Ort. Fiyat: ${grup_1['price'].mean():,.0f}")
print("-" * 30)

print("\n--- GRUP 0 (Clean Title Olmayan) İçindeki En Pahalı 10 Marka Ortalaması ---")
# Grup 0'da olup fiyatı en yüksek olan markalar hangileri?
top_brands_missing_title = grup_0.groupby('brand')['price'].mean().sort_values(ascending=False).head(10)
print(top_brands_missing_title)

print("\n--- GRUP 0 vs GRUP 1 Marka Dağılımı ---")
# Lüks markalarda Clean Title boş olma oranı nedir?
luxury_brands = ['Bugatti', 'Rolls-Royce', 'Lamborghini', 'McLaren', 'Ferrari', 'Bentley']
for brand in luxury_brands:
    total = len(df[df['brand'] == brand])
    if total > 0:
        missing = len(df[(df['brand'] == brand) & (df['clean_title_binary'] == 0)])
        print(f"{brand}: Toplam {total} araç, {missing} tanesinin Clean Title'ı BOŞ (%{missing/total*100:.1f})")

# Görsel Kanıt
plt.figure(figsize=(10, 6))
sns.boxplot(x='clean_title_binary', y='price', data=df)
plt.yscale('log')
plt.title('Paradoks Kontrolü: 0 Olanların Fiyat Aralığı Neden Yüksek?')
plt.show()

# Hocana ve Sınıfa Şunu Söylemelisin:
# 
# Dürüstlük: "İlk analizde Clean Title olmayan araçların daha pahalı olduğunu gördük, bu mantıksızdı."
# 
# Veri Dedektifliği: "Veriyi deşifre ettiğimde gördüm ki; Bugatti, Rolls-Royce gibi ultra lüks araç ilanlarında satıcılar 'Clean Title' bilgisini girmemiş (NaN bırakmış). Biz NaN değerleri 0 kabul edince, bu pahalı arabalar 'Belgesiz' grubunun ortalamasını yukarı çekmiş."
# 
# Çözüm/Yorum: "Bu yüzden modellemede clean_title sütunu, fiyatı tahmin ederken beklediğimiz 'düşürücü' etkiyi yaratmayabilir, aksine lüks araç sinyali gibi davranabilir. Veri biliminde buna 'Proxy Variable' (Vekil Değişken) denir."


display(df.head())

# 2. YAŞ HESAPLAMA (FEATURE ENGINEERING)
# ========================================================
current_year = 2025 # Projeyi yaptığımız yıl

# Yaş hesabı
df['age'] = current_year - df['model_year']

print(f"--- Araç Yaşı Analizi (Referans Yıl: {current_year}) ---")
print(df[['model_year', 'age']].head())
print("\n")

# İstatistikler
print("--- Yaş İstatistikleri ---")
print(df['age'].describe())


# 3. GÖRSELLEŞTİRME VE KARŞILAŞTIRMA
# ========================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Yaş Dağılımı
sns.histplot(df['age'], bins=20, kde=True, color='purple', ax=axes[0])
axes[0].set_title('Araç Yaş Dağılımı')
axes[0].set_xlabel('Yaş')

# Yaş vs Fiyat İlişkisi (Line Plot)
# Yaş arttıkça ortalama fiyat nasıl değişiyor?
age_price = df.groupby('age')['price'].mean().reset_index()
sns.lineplot(x='age', y='price', data=age_price, marker='o', color='red', ax=axes[1])
axes[1].set_title('Araç Yaşı ve Ortalama Fiyat İlişkisi')
axes[1].set_ylabel('Ortalama Fiyat ($)')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Analiz ve Beklentiler 📉
# Fiyat Eğrisi: Sağdaki grafikte (Line Plot), yaş arttıkça fiyatın hızla düştüğünü göreceğiz. Ancak 30-40 yaşından sonra grafik tekrar yukarı dönebilir!
# 
# Neden? Çünkü o araçlar artık "Eski" değil, "Klasik / Antika" sınıfına girer ve fiyatları artmaya başlar. Bunu yakalayabilirsek sunumda harika bir detay olur.
# 
# Sütun Değişimi: Artık modelimiz 2020 yılını değil, 5 yaşını öğrenecek. Bu, modelin genelleme yeteneğini artırır.


# 4. MODEL_YEAR SÜTUNUNU KALDIRMA
# ========================================================
# Artık 'age' bilgisi elimizde olduğu için 'model_year' gereksiz (Multicollinearity yaratır).
df.drop('model_year', axis=1, inplace=True)

print("--- İşlem Tamamlandı ---")
print("'model_year' sütunu silindi, yerine 'age' sütunu eklendi.")
print(f"Yeni Sütun Listesi: {df.columns.tolist()}")

display(df.head())

# ## OHE oncesi analiz


# ========================================================
# 1. KARDİNALİTE (ÇEŞİTLİLİK) ANALİZİ
# ========================================================
# Hangi kategorik sütunda kaç farklı değer var?

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("--- Kategorik Sütunlardaki Benzersiz Değer Sayısı ---")
cardinality = df[categorical_cols].nunique().sort_values(ascending=False)
print(cardinality)

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.barplot(x=cardinality.values, y=cardinality.index, palette='Reds_r')
plt.title('Hangi Sütun One-Hot Encoding İçin Tehlikeli?')
plt.xlabel('Benzersiz Değer Sayısı (Unique Count)')
for i, v in enumerate(cardinality.values):
    plt.text(v + 10, i, str(v), va='center')
plt.show()

# ========================================================
# 2. DETAYLI İNCELEME (Brand ve Renkler)
# ========================================================
print("\n--- Brand (Marka) Analizi: İlk 20 ---")
print(df['brand'].value_counts().head(20))

print("\n--- Exterior Color (Dış Renk) Analizi: İlk 20 ---")
print(df['ext_col'].value_counts().head(20))

print("\n--- Model Analizi: İlk 10 ---")
print(df['model'].value_counts().head(10))

# ========================================================
# 3. RENK SADELEŞTİRME (COLOR REDUCTION)
# ========================================================
# Hedef: 'Moonlight Blue', 'Deep Blue' -> 'Blue'

def clean_color(color):
    if pd.isna(color): return 'Other'
    color = str(color).lower()
    
    common_colors = ['black', 'white', 'gray', 'silver', 'blue', 'red', 'brown', 'green', 'beige', 'gold', 'yellow', 'orange']
    
    for c in common_colors:
        if c in color:
            return c.capitalize()
            
    return 'Other' # Hiçbirine girmiyorsa (Örn: 'Unknown', 'Pewter')

df['ext_col_clean'] = df['ext_col'].apply(clean_color)
df['int_col_clean'] = df['int_col'].apply(clean_color)

# Eski kirli sütunları at
df.drop(['ext_col', 'int_col'], axis=1, inplace=True)

print("--- Renk Sadeleştirme Sonrası Dağılım ---")
print(df['ext_col_clean'].value_counts())

# ========================================================
# 3. MARKA GRUPLAMA (RARE LABELS)
# ========================================================
# Az bulunan markaları 'Other' kategorisine alalım.

threshold = 20 # Veri setinde 20'den az bulunan markalar
brand_counts = df['brand'].value_counts()
rare_brands = brand_counts[brand_counts < threshold].index

df['brand_clean'] = df['brand'].apply(lambda x: 'Other' if x in rare_brands else x)
df.drop('brand', axis=1, inplace=True) # Eski sütunu at

print(f"\n--- Marka Gruplama Sonrası ({len(rare_brands)} marka 'Other' yapıldı) ---")
print(df['brand_clean'].value_counts().tail())

# ========================================================
# 3. KİRLİ SÜTUNLARI TOPLU SİLME (GARANTİ YÖNTEM)
# ========================================================
# Hangi sütunları istemiyoruz? Hepsini listeye yazıp tek seferde siliyoruz.
dirty_cols_to_drop = [
    'model_year', 'fuel_type', 'engine', 'transmission', 
    'accident', 'clean_title', 'ext_col', 'int_col', 
    'brand', 'model'
]

# errors='ignore' parametresi, eğer sütun zaten silinmişse hata vermemesini sağlar.
df.drop(columns=dirty_cols_to_drop, axis=1, inplace=True, errors='ignore')

print(f"Temizlik Sonrası Sütunlar (Encoding Öncesi): {df.columns.tolist()}")

# ========================================================
# 4. ONE-HOT ENCODING (INT DÖNÜŞÜMLÜ)
# ========================================================

categorical_cols = ['fuel_category', 'transmission_type', 'ext_col_clean', 'int_col_clean', 'brand_clean']

# dtype=int PARAMETRESİ BURADA ÇOK ÖNEMLİ! (True/False yerine 1/0 yapar)
df_final = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# ========================================================
# 5. FİNAL KONTROL
# ========================================================
print("\n" + "="*50 + "\n")
print("--- FINAL VERİ SETİ KONTROLÜ ---")
print(f"Sütun Sayısı: {df_final.shape[1]}")

print("\n--- Veri Tipleri (Dtypes) ---")
# Value counts ile kaç tane int, kaç tane float var bakalım. Object ve Bool OLMAMALI.
print(df_final.dtypes.value_counts())

print("\n--- İlk 5 Satır (Örnek) ---")
display(df_final.head())

# Kategorik Veri Kaldı mı? (Sağlama)
object_cols = df_final.select_dtypes(include=['object', 'bool']).columns.tolist()
if len(object_cols) == 0:
    print("\n✅ BAŞARILI: Veri setinde hiç 'Object' veya 'Bool' kalmadı. Hepsi sayısal (int/float).")
    # Veriyi kaydet
    df_final.to_csv('cleaned_used_cars_final.csv', index=False)
    print("✅ Dosya 'cleaned_used_cars_final.csv' olarak kaydedildi.")
else:
    print(f"\n❌ HATA: Şu sütunlar hala kategorik veya boolean: {object_cols}")

# 1. VERİYİ YÜKLE (Önceki aşamada kaydettiğimiz final verisi)
# ========================================================
# Eğer önceki adımda dosya kaydedilmediyse, hafızadaki df_final kullanılır.
# Biz burada df_final'in hazır olduğunu varsayıyoruz.
# (Simülasyon için df_final'i tekrar oluşturacak kod parçası burada yer almaz, 
# doğrudan df_final üzerinden devam edilir. Eğer df_final yoksa hata verir.)

try:
    df = df_final.copy() # Orijinal veriyi bozmamak için kopya alıyoruz
    print("Veri seti hafızadan alındı.")
except NameError:
    # Eğer hafızada yoksa dosyadan okumayı dene (Önceki adımları çalıştırdıysan bu çalışır)
    try:
        df = pd.read_csv('cleaned_used_cars_final.csv')
        print("Veri seti dosyadan yüklendi.")
    except:
        print("HATA: Lütfen önce 'veri_hazirlik_final_corrected.py' dosyasını çalıştırın!")

print(f"İşlem Başlangıcı Veri Boyutu: {df.shape}")

# ========================================================
# 2. AYKIRI VERİ ANALİZİ VE TEMİZLİĞİ (OUTLIER HANDLING)
# ========================================================
# Yöntem: IQR (Interquartile Range)
# Özellikle 'price' sütunundaki aşırı uçlar modeli bozar.

print("\n--- 1. Aykırı Veri Temizliği (Fiyat Odaklı) ---")

def remove_outliers_iqr(dataframe, col_name, multiplier=1.5):
    Q1 = dataframe[col_name].quantile(0.25)
    Q3 = dataframe[col_name].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    print(f"{col_name} için Alt Sınır: {lower_bound:.2f}, Üst Sınır: {upper_bound:.2f}")
    
    # Sınırlar dışındakileri at
    df_filtered = dataframe[~((dataframe[col_name] < lower_bound) | (dataframe[col_name] > upper_bound))]
    return df_filtered

# Fiyat için temizlik (Multiplier=3 yaparak çok agresif silmeyelim, sadece aşırı uçları silelim)
# Çünkü lüks arabalar doğal olarak pahalıdır, hepsini silmek istemeyiz.
df_clean = remove_outliers_iqr(df, 'price', multiplier=3.0)

removed_count = len(df) - len(df_clean)
print(f"Temizlenen Aykırı Veri Sayısı: {removed_count} (Toplamın %{removed_count/len(df)*100:.2f}'si)")

# Görsel Karşılaştırma (Önce - Sonra)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.boxplot(x=df['price'], ax=axes[0], color='red').set_title('Temizlik Öncesi Fiyat (Outliers)')
sns.boxplot(x=df_clean['price'], ax=axes[1], color='green').set_title('Temizlik Sonrası Fiyat')
plt.show()

# Veriyi güncelle
df = df_clean.copy()

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
# ========================================================
# 3. NORMALİZASYON / STANDARDİZASYON (SCALING)
# ========================================================
# Yöntem: StandardScaler (Z-Score Normalization)
# Formül: z = (x - u) / s
# Hedef Sütunlar: Sürekli sayısal değişkenler (Binary/0-1 olanlara dokunmuyoruz)

print("\n--- 2. Özellik Ölçeklendirme (Standardization) ---")

scale_cols = ['milage', 'age', 'horsepower', 'engine_volume', 'cylinders']
print(f"Ölçeklenecek Sütunlar: {scale_cols}")

# Scaler nesnesini oluştur
scaler = StandardScaler()

# Ölçeklemeyi uygula
# Not: Normalde Train/Test split yaptıktan sonra fit etmek daha doğrudur (Data Leakage önlemek için).
# Ancak proje raporunda "Preprocessing" başlığı altında göstermek için burada tüm veriye uyguluyoruz.
# Sunumda: "Veri sızıntısını önlemek için bu işlemi pipeline içinde de yapabiliriz" notunu düşebilirsin.

df[scale_cols] = scaler.fit_transform(df[scale_cols])

print("Ölçeklendirme tamamlandı. İlk 5 satırın istatistikleri (Ortalama ~0, Std ~1 olmalı):")
print(df[scale_cols].describe().round(2).loc[['mean', 'std', 'min', 'max']])


# ========================================================
# 4. ÖZELLİK SEÇİMİ VE ÖNEMİ (FEATURE IMPORTANCE)
# ========================================================
# Yöntem: Random Forest Regressor kullanarak özelliklerin önem skorlarını çıkarma.

print("\n--- 3. Özellik Önem Analizi (Feature Importance) ---")

# Hedef ve Özellikleri Ayır
X = df.drop('price', axis=1)
y = df['price']

# Modeli Kur (Hızlı sonuç için az ağaçlı)
rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

# Önem Skorlarını Çek
importances = rf_model.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

# Raporlama
print("\nEn Önemli 10 Özellik:")
print(feature_imp_df.head(10))

# Görselleştirme (Bar Plot)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(20), palette='viridis')
plt.title('Araba Fiyatını Etkileyen En Önemli Özellikler')
plt.xlabel('Önem Skoru (Feature Importance)')
plt.ylabel('Özellikler')
plt.show()

# ========================================================
# 5. FEATURE SELECTION (GEREKSİZLERİ ATMA) neden PCA kullanmadık? XAI için
# ========================================================
# Eşiğin altında kalan (etkisiz) özellikleri atalım.
threshold = 0.001 # %0.1'den az etkisi olanlar çöp
unimportant_features = feature_imp_df[feature_imp_df['Importance'] < threshold]['Feature'].tolist()

print(f"\nAtılacak Etkisiz Özellik Sayısı: {len(unimportant_features)}")
print(f"Örnek Atılacaklar: {unimportant_features[:5]}")

df_final_processed = df.drop(columns=unimportant_features)

print(f"\n--- İŞLEM TAMAMLANDI ---")
print(f"Son Veri Seti Boyutu: {df_final_processed.shape}")
print("Veri artık modellenmeye %100 hazır.")

# Son hali kaydet (Modelleme aşaması için)
df_final_processed.to_csv('processed_used_cars_ready_for_model.csv', index=False)