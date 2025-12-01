import pandas as pd
import numpy as np
df = pd.read_csv("/Users/hasantayfur/Downloads/used_cars.csv")
print(df.head())

# Veri boyutu (satır ve sütun sayısı)
print("\n--- Veri Boyutu (Satır, Sütun) ---")
print(df.shape)

# Sütun bilgileri (Veri Türleri)
print("\n--- Sütun Bilgileri ve Veri Tipleri ---")
print(df.info())

# Sayısal sütunların istatistiksel özeti
print("\n--- Sayısal Özelliklerin İstatistiksel Özeti ---")
print(df.describe().T)

# Eksik veri kontrolü
print("\n--- Eksik Veri Sayıları ---")
print(df.isnull().sum())


# 'clean_title' kolonunun ilk analizini yapın
print("--- 'clean_title' kolonunun ilk analizi ---")
print(df['clean_title'].value_counts(dropna=False))
print("\nBoş değerlerin sayısı:", df['clean_title'].isnull().sum())

 #Önce kolonun object (metin) tipinde olduğundan emin olalım
df['price'] = df['price'].astype(str)

# Dolar işaretini ($) ve virgülü (,) kaldır
df['price'] = df['price'].str.replace('$', '', regex=False)
df['price'] = df['price'].str.replace(',', '', regex=False)

# Bazı değerler ""$36,000"" şeklinde çift tırnak içinde olabilir, onları da temizleyelim
df['price'] = df['price'].str.replace('"', '', regex=False)

# Temizlenmiş kolonun veri tipini sayısal (integer) yap
# errors='coerce' parametresi, dönüştürme sırasında hata olursa o değeri boş (NaN) yapar
df['price'] = pd.to_numeric(df['price'], errors='coerce')

print("'price' kolonu başarıyla temizlendi ($, virgül ve tırnak işaretleri kaldırıldı, sayısal tipe dönüştürüldü).")


# --- Sonucu Kontrol Etme ---
print("\n--- Temizlenmiş Verilerden Örnekler ---")
print(df[['clean_title', 'price']].head())


# Boş (NaN) değerleri 0 ile değiştirin
df['clean_title'].fillna(0, inplace=True)

# Adım 2: 'clean_title' kolonundaki 'Yes' değerlerini 1 ile değiştirin
# .replace() metodu ile 'Yes' string'ini integer 1 ile değiştiriyoruz.
df['clean_title'] = df['clean_title'].replace('Yes', 1)
print("'clean_title' kolonundaki 'Yes' değerleri '1' ile değiştirildi.")


# 'clean_title' kolonunu boş değerler doldurulduktan sonra tekrar analiz edin
print("\n--- Boş değerler 0 ile doldurulduktan sonra 'clean_title' kolonunun analizi ---")
print(df['clean_title'].value_counts(dropna=False))
print("\nBoş değerlerin sayısı (güncellenmiş):", df['clean_title'].isnull().sum())


#Adım 3: 'milage' Kolonunu Temizleme (YENİ EKLENEN KISIM) ---
# Önce kolonun object (metin) tipinde olduğundan emin olalım
df['milage'] = df['milage'].astype(str)

# " mi." ifadesini ve virgülü (,) kaldır
df['milage'] = df['milage'].str.replace(' mi.', '', regex=False)
df['milage'] = df['milage'].str.replace(',', '', regex=False)

# Temizlenmiş kolonun veri tipini sayısal (integer) yap
df['milage'] = pd.to_numeric(df['milage'], errors='coerce')
print("3. 'milage' kolonu temizlendi (' mi.' ve virgül kaldırıldı).")


# --- Sonucu Kontrol Etme ---
print("\n--- Temizlenmiş Verilerden Örnekler ---")
print(df[['clean_title', 'price', 'milage']].head())

# Veri tiplerini kontrol edelim
print("\n--- Kolonların Son Veri Tipleri ---")
print(df[['clean_title', 'price', 'milage']].dtypes)


#  Adım 4: 'accident' Kolonunu Makine Öğrenmesine Hazır Hale Getirme (YENİ EKLENEN KISIM) ---
# Metin içeren bu kolonu 0 ve 1'lere dönüştüreceğiz.
# 'None reported' değerini 0, diğer tüm durumları (kaza raporu olanları) 1 yapacağız.
# np.where(koşul, doğruysa_bu, yanlışsa_bu) yapısını kullanmak oldukça etkilidir.
df['accident_binary'] = np.where(df['accident'] == 'None reported', 0, 1)

# Orijinal 'accident' kolonuna artık ihtiyacımız yoksa kaldırabiliriz
# df = df.drop('accident', axis=1)

print("4. 'accident' kolonu 'accident_binary' adıyla sayısal formata dönüştürüldü (None reported -> 0, Diğerleri -> 1).")








# --- Sonucu Kontrol Etme ---
print("\n--- Temizlenmiş ve Dönüştürülmüş Verilerden Örnekler ---")
# Yeni oluşturulan kolonu da ekleyerek bakalım
print(df[['clean_title', 'price', 'milage', 'accident', 'accident_binary']].head())

# Yeni kolonun değer dağılımını kontrol edelim
print("\n--- 'accident_binary' Kolonunun Değer Dağılımı ---")
print(df['accident_binary'].value_counts())

# Adım 5: Eski 'accident' kolonunu kaldır (YENİ EKLENEN KISIM)
df.drop('accident', axis=1, inplace=True)
# axis=1 -> Kolon olduğunu belirtir
# inplace=True -> Değişikliği DataFrame üzerinde kalıcı yapar
print("5. Orijinal 'accident' kolonu veri setinden kaldırıldı.")

















# Temizlenmiş veri setini yeni bir CSV dosyasına kaydedin
df.to_csv('cleaned_used_cars.csv', index=False)

print("\n'cleaned_used_cars.csv' dosyası başarıyla oluşturuldu.")

df = pd.read_csv("cleaned_used_cars.csv")
print(df.head())

