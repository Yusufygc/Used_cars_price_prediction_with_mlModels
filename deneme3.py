import pandas as pd
import numpy as np

# Dosyanın adını ve yolunu doğru girdiğinizden emin olun
file_path = '/Users/hasantayfur/Downloads/used_cars.csv'

try:
    df = pd.read_csv(file_path)
    print("--- Veri Temizleme ve Gruplama İşlemi Başladı ---")
    
    # Önceki adımları tekrar uygulayalım
    df['clean_title'] = df['clean_title'].replace('Yes', 1)
    df['clean_title'].fillna(0, inplace=True)
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace('[$,"]', '', regex=True), errors='coerce')
    df['milage'] = pd.to_numeric(df['milage'].astype(str).str.replace(' mi.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
    df['accident_binary'] = np.where(df['accident'] == 'None reported', 0, 1)
    df.drop('accident', axis=1, inplace=True)

    # --- 'fuel_type' Kolonu Analiz ve Gruplama ---
    print("\n--- 'fuel_type' Kolonu Analizi ---")
    print(f"Boş 'fuel_type' verilerinin sayısı: {df['fuel_type'].isnull().sum()}")
    print("\n--- Orijinal Yakıt Türü Dağılımı ---")
    print(df['fuel_type'].value_counts())
    
    # Boş yakıt türü verilerini 'Other' olarak dolduralım
    df['fuel_type'].fillna('Other', inplace=True)

    # Gruplama için bir fonksiyon tanımlayalım
    def categorize_fuel(fuel_type):
        fuel_type = str(fuel_type).lower() # Büyük/küçük harf sorununu önlemek için
        if 'hybrid' in fuel_type:
            return 'Hybrid'
        elif 'electric' in fuel_type:
            return 'Electric'
        elif 'gasoline' in fuel_type or 'flex fuel' in fuel_type or 'diesel' in fuel_type:
            return 'Fuel'
        else:
            return 'Other' # Diğer durumlar için (Hydrogen vb.)

    # Fonksiyonu uygulayarak yeni bir kolon oluşturalım
    df['fuel_category'] = df['fuel_type'].apply(categorize_fuel)
    
    print("\n--- Gruplanmış Yeni Yakıt Kategorileri Dağılımı ---")
    print(df['fuel_category'].value_counts())

    # One-Hot Encoding işlemini yeni 'fuel_category' kolonuna uygulayalım
    fuel_dummies = pd.get_dummies(df['fuel_category'], prefix='fuel', drop_first=True)
    df = pd.concat([df, fuel_dummies], axis=1)
    
    # Artık orijinal 'fuel_type' ve geçici 'fuel_category' kolonlarını silebiliriz
    df.drop(['fuel_type', 'fuel_category'], axis=1, inplace=True)
    
    print("\n'fuel_type' kolonu başarıyla 'Fuel', 'Hybrid', 'Electric' olarak gruplandı ve dönüştürüldü.")

    # --- Sonucu Kontrol Etme ---
    print("\n--- Nihai Veri Setindeki Kolonlar ---")
    print(df.columns)

    output_file = 'cleaned_used_cars_for_ML_final.csv'
    df.to_csv(output_file, index=False)
    print(f"\nİşlem tamamlandı! Nihai veri '{output_file}' dosyasına kaydedildi.")

except FileNotFoundError:
    print(f"HATA: '{file_path}' dosyası bulunamadı.")