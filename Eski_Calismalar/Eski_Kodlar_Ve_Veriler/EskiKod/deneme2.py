import pandas as pd
import numpy as np

# Dosyanın adını ve yolunu doğru girdiğinizden emin olun
file_path = '/Users/hasantayfur/Downloads/used_cars.csv'

try:
    # CSV dosyasını bir DataFrame'e yükleyin
    df = pd.read_csv(file_path)

    print("--- Veri Temizleme ve Hazırlama İşlemi Başladı ---")

    # Adım 1: 'clean_title' kolonunu temizle
    df['clean_title'].fillna(0, inplace=True)
    df['clean_title'] = df['clean_title'].replace('Yes', 1)
    print("1. 'clean_title' kolonu temizlendi.")

    # Adım 2: 'price' kolonunu temizle
    df['price'] = df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace('"', '', regex=False)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    print("2. 'price' kolonu temizlendi.")
    
    # Adım 3: 'milage' kolonunu temizle
    df['milage'] = df['milage'].astype(str).str.replace(' mi.', '', regex=False).str.replace(',', '', regex=False)
    df['milage'] = pd.to_numeric(df['milage'], errors='coerce')
    print("3. 'milage' kolonu temizlendi.")

    # Adım 4: 'accident' kolonunu sayısal formata dönüştür ve eskisini sil
    df['accident_binary'] = np.where(df['accident'] == 'None reported', 0, 1)
    df.drop('accident', axis=1, inplace=True)
    print("4. 'accident' kolonu sayısal formata dönüştürüldü ve orijinali silindi.")

    # Adım 5: 'fuel_type' kolonunu analiz et ve dönüştür (YENİ EKLENEN KISIM)
    print("5. 'fuel_type' kolonu analiz ediliyor ve dönüştürülüyor...")
    
    # Öncelikle kolondaki kategori dağılımına bakalım
    print("\n--- 'fuel_type' Kolonundaki Değerlerin Dağılımı ---")
    print(df['fuel_type'].value_counts())
    
    # One-Hot Encoding uygula
    # prefix -> Yeni kolon adlarının başına 'fuel' kelimesini ekler (fuel_Gasoline gibi)
    # drop_first=True -> Çoklu doğrusallığı (multicollinearity) önlemek için ilk kategoriyi atar. Bu, model performansı için iyi bir pratiktir.
    fuel_dummies = pd.get_dummies(df['fuel_type'], prefix='fuel', drop_first=True)
    
    # Oluşturulan yeni dummy kolonları ana DataFrame'e ekle
    df = pd.concat([df, fuel_dummies], axis=1)
    
    # Artık orijinal 'fuel_type' kolonuna ihtiyacımız kalmadı, silelim
    df.drop('fuel_type', axis=1, inplace=True)
    
    print("\n   'fuel_type' kolonu One-Hot Encoding ile dönüştürüldü ve orijinali silindi.")


    # --- Sonucu Kontrol Etme ---
    print("\n--- Nihai Veri Setindeki Kolonlar ---")
    print(df.columns)
    
    print("\n--- Dönüşüm Sonrası Veriden Bir Örnek ---")
    print(df.head())

    # Değişiklikleri yeni bir CSV dosyasına kaydet
    output_file = 'cleaned_used_cars_for_ML.csv'
    df.to_csv(output_file, index=False)

    print(f"\nİşlem tamamlandı! Makine öğrenmesine hazır nihai veri '{output_file}' dosyasına kaydedildi.")

except FileNotFoundError:
    print(f"HATA: '{file_path}' dosyası bulunamadı.")
    print("Lütfen dosya adının doğru olduğundan ve dosyanın kodun çalıştığı dizinde bulunduğundan emin olun.")