import pandas as pd
import numpy as np
import re
import os

# Genel Ayarlar
INPUT_FILE_NAME = 'ml/used_cars.csv'
OUTPUT_FILE_NAME = 'ml/GPT3.1_cleaned_and_engineered_cars.csv'
CURRENT_YEAR = 2025

# ------------------------------------------------
# 1. Veri Yükleme
# ------------------------------------------------
def load_data(file_name):
    try:
        df = pd.read_csv(file_name)
        print(f"'{file_name}' dosyası başarıyla yüklendi. Satır sayısı: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"HATA: '{file_name}' dosyası bulunamadı.")
        return None


# ------------------------------------------------
# 2. FUEL TYPE Analizi + Binary Encoding
# ------------------------------------------------
def analyze_fuel_type(df):
    """
    'fuel_type' kolonunu analiz eder, temizler, encoding uygular.
    - fuel_group oluşturur (ama en sonda kaldırılır)
    - One-hot encoding sütunlarını binary (1/0) formata dönüştürür
    - 'fuel_type' ve 'fuel_group' kolonlarını kaldırır
    """
    print("\n--- Adım: Fuel Type Analizi ve Encoding ---")

    print("\nYakıt Türü Dağılımı:")
    print(df['fuel_type'].value_counts(dropna=False))
    print(f"\nEksik değer sayısı: {df['fuel_type'].isna().sum()}")

    df['fuel_type'] = df['fuel_type'].fillna('Unknown')

    def classify_fuel_type(ftype):
        ftype = str(ftype).lower()
        if any(x in ftype for x in ['gas', 'petrol']):
            return 'Gasoline'
        elif 'diesel' in ftype:
            return 'Diesel'
        elif 'hybrid' in ftype and 'plug' not in ftype:
            return 'Hybrid'
        elif 'plug' in ftype or 'electric' in ftype:
            return 'Electric'
        elif 'flex' in ftype:
            return 'Flex Fuel'
        elif ftype == 'unknown':
            return 'Unknown'
        else:
            return 'Other'

    df['fuel_group'] = df['fuel_type'].apply(classify_fuel_type)

    before_len = len(df)
    df['fuel_group'] = df['fuel_group'].replace('Unknown', 'Electric')
    df = df[df['fuel_group'] != 'Other']
    after_len = len(df)

    print(f"\n{before_len - after_len} satır 'Other' kategorisinden silindi.")
    print("Tüm 'Unknown' değerler 'Electric' kategorisine eklendi.")

    # One-hot encoding ve binary dönüşüm
    fuel_dummies = pd.get_dummies(df['fuel_group'], prefix='fuel', drop_first=False).astype(int)
    df = pd.concat([df, fuel_dummies], axis=1)

    # fuel_type ve fuel_group kaldır
    for col in ['fuel_type', 'fuel_group']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print(f"-> '{col}' kolonu kaldırıldı.")

    print("\nYeni Binary Fuel Encoding Sütunları (1/0):")
    print(list(fuel_dummies.columns))

    return df


# ------------------------------------------------
# 3. TRANSMISSION Analiz ve Temizlik
# ------------------------------------------------
def analyze_transmission(df):
    print("\n--- Adım: Transmission Kolonu Analizi ---")
    if 'transmission' not in df.columns:
        print("HATA: 'transmission' kolonu bulunamadı.")
        return df

    total = len(df)
    missing = df['transmission'].isna().sum()
    print(f"\nToplam kayıt sayısı: {total}")
    print(f"Eksik değer sayısı: {missing} ({(missing/total)*100:.2f}%)")

    print("\nTransmission Tipi Dağılımı:")
    print(df['transmission'].value_counts(dropna=False))
    return df


def clean_transmission(df):
    """
    'transmission' kolonunu temizler, gruplaştırır, Unknown'ları siler,
    Electric araçların vites tipini 'Single-Speed' olarak ayarlar,
    One-Hot Encoding uygular ve sayısal hale getirir.
    """
    print("\n--- Adım: Transmission Kolonu Temizliği ve Encoding ---")

    def classify_transmission(t):
        if pd.isna(t):
            return 'Unknown'
        t = str(t).lower().strip()
        if any(x in t for x in ['a/t', 'automatic', 'auto,', 'at', 'aut']):
            return 'Automatic'
        elif any(x in t for x in ['m/t', 'manual', 'mt']):
            return 'Manual'
        elif 'cvt' in t:
            return 'CVT'
        elif any(x in t for x in ['dual', 'dct', 'auto-shift', 'shift mode']):
            return 'Semi-Automatic'
        elif any(x in t for x in ['single-speed', '1-speed', '2-speed']):
            return 'Single-Speed'
        elif any(x in t for x in ['variable', '–', 'scheduled', 'overdrive', 'f']):
            return 'Unknown'
        else:
            return 'Unknown'

    # 1️⃣ Transmission gruplarını oluştur
    df['transmission_group'] = df['transmission'].apply(classify_transmission)

    # 2️⃣ Elektrikli araçlar için "Single-Speed" ayarı
    if 'fuel_Electric' in df.columns:
        electric_before = df['fuel_Electric'].sum()
        df.loc[df['fuel_Electric'] == 1, 'transmission_group'] = 'Single-Speed'
        print(f"\n{electric_before} elektrikli araç için transmission_group = 'Single-Speed' olarak ayarlandı.")

    # 3️⃣ Unknown satırları sil
    before_len = len(df)
    df = df[df['transmission_group'] != 'Unknown']
    after_len = len(df)
    print(f"{before_len - after_len} satır 'Unknown' transmission nedeniyle silindi.")

    # 4️⃣ One-Hot Encoding ve binary dönüşüm
    trans_dummies = pd.get_dummies(df['transmission_group'], prefix='trans', drop_first=False).astype(int)
    df = pd.concat([df, trans_dummies], axis=1)

    # 5️⃣ Orijinal kolonları kaldır
    for col in ['transmission', 'transmission_group']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print(f"-> '{col}' kolonu kaldırıldı.")

    # 6️⃣ Kontrol çıktısı
    print("\nYeni Transmission Encoding Sütunları (1/0):")
    print(list(trans_dummies.columns))
    print(df[list(trans_dummies.columns)].head())

    return df



# ------------------------------------------------
# 4. Diğer Temizlik ve Özellik Fonksiyonları
# ------------------------------------------------
def clean_price_and_milage(df):
    df['price'] = pd.to_numeric(df['price'].astype(str)
                                .str.replace('$', '')
                                .str.replace(',', '')
                                .str.replace('"', ''), errors='coerce')
    df['milage'] = pd.to_numeric(df['milage'].astype(str)
                                 .str.replace(' mi.', '')
                                 .str.replace(',', ''), errors='coerce')
    print("-> 'price' ve 'milage' kolonları temizlendi.")
    return df


def prepare_binary_columns(df):
    df['clean_title'].fillna(0, inplace=True)
    df['clean_title'] = df['clean_title'].replace('Yes', 1)
    df['clean_title'] = pd.to_numeric(df['clean_title'], errors='coerce').fillna(0).astype(int)
    df['accident_binary'] = np.where(df['accident'] == 'None reported', 0, 1)
    df.drop('accident', axis=1, inplace=True)
    print("-> Binary sütunlar düzenlendi.")
    return df


def extract_engine_features(df):
    """'engine' sütununu HP, hacim (L) ve silindir sayısına ayırır."""
    def extract_horsepower(s):
        m = re.search(r'(\d+\.?\d*)\s*HP', str(s))
        return float(m.group(1)) if m else np.nan

    def extract_volume(s):
        m = re.search(r'(\d+\.?\d*)\s*(?:L|Liter)', str(s))
        return float(m.group(1)) if m else np.nan

    def extract_cylinders(s):
        if pd.isna(s):
            return np.nan
        s = str(s)
        if m := re.search(r'(\d+)\s*(?:Cylinder|Cyl)', s):
            return int(m.group(1))
        if v := re.search(r'V(\d+)', s):
            return int(v.group(1))
        return np.nan

    df['horsepower'] = df['engine'].apply(extract_horsepower)
    df['engine_volume_L'] = df['engine'].apply(extract_volume)
    df['cylinder_count'] = df['engine'].apply(extract_cylinders)

    # Elektrikli araçlar (fuel_Electric == 1)
    if 'fuel_Electric' in df.columns:
        is_electric = (df['fuel_Electric'] == 1)
        df.loc[is_electric, ['engine_volume_L', 'cylinder_count']] = 0

    df['horsepower'].fillna(df['horsepower'].median(), inplace=True)
    df['engine_volume_L'].fillna(df['engine_volume_L'].median(), inplace=True)
    df['cylinder_count'].fillna(df['cylinder_count'].median(), inplace=True)
    df['cylinder_count'] = df['cylinder_count'].astype(int)

    df.drop('engine', axis=1, inplace=True)
    print("-> 'engine' kolonu sayısal özelliklere dönüştürüldü.")
    return df


def calculate_car_age(df, current_year):
    df['car_age'] = current_year - df['model_year']
    df.drop('model_year', axis=1, inplace=True)
    print("-> 'car_age' hesaplandı.")
    return df


# ------------------------------------------------
# 5. ANA ÇALIŞMA AKIŞI
# ------------------------------------------------
def main_workflow():
    df = load_data(INPUT_FILE_NAME)
    if df is None:
        return

    print("\n--- Adım 1: Price & Milage ---")
    df = clean_price_and_milage(df)

    print("\n--- Adım 2: Binary Kolonlar ---")
    df = prepare_binary_columns(df)

    print("\n--- Adım 3: Fuel Type Analizi ---")
    df = analyze_fuel_type(df)

    print("\n--- Adım 4: Transmission Analizi ---")
    df = analyze_transmission(df)

    print("\n--- Adım 4.5: Transmission Temizliği ---")
    df = clean_transmission(df)

    print("\n--- Adım 5: Engine Özellikleri ---")
    df = extract_engine_features(df)

    print("\n--- Adım 6: Araç Yaşı ---")
    df = calculate_car_age(df, CURRENT_YEAR)

    print("\n--- Adım 7: Kaydetme ---")
    df.to_csv(OUTPUT_FILE_NAME, index=False)
    print(f"\n✅ İşlem tamamlandı. '{OUTPUT_FILE_NAME}' dosyasına kaydedildi.")


# Ana fonksiyon çağrısı
if __name__ == "__main__":
    main_workflow()
