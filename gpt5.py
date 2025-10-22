import pandas as pd
import numpy as np
import re
import os

# Genel Ayarlar
INPUT_FILE_NAME = 'ml/used_cars.csv'
OUTPUT_FILE_NAME = 'ml/GPT5_cleaned_and_engineered_cars.csv'
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

    fuel_dummies = pd.get_dummies(df['fuel_group'], prefix='fuel', drop_first=False).astype(int)
    df = pd.concat([df, fuel_dummies], axis=1)

    for col in ['fuel_type', 'fuel_group']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print(f"-> '{col}' kolonu kaldırıldı.")

    print("\nYeni Binary Fuel Encoding Sütunları (1/0):")
    print(list(fuel_dummies.columns))
    return df


# ------------------------------------------------
# 3. TRANSMISSION Temizleme + Encoding
# ------------------------------------------------
def clean_transmission(df):
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

    df['transmission_group'] = df['transmission'].apply(classify_transmission)

    if 'fuel_Electric' in df.columns:
        electric_before = df['fuel_Electric'].sum()
        df.loc[df['fuel_Electric'] == 1, 'transmission_group'] = 'Single-Speed'
        print(f"\n{electric_before} elektrikli araç için transmission_group = 'Single-Speed' olarak ayarlandı.")

    before_len = len(df)
    df = df[df['transmission_group'] != 'Unknown']
    after_len = len(df)
    print(f"{before_len - after_len} satır 'Unknown' transmission nedeniyle silindi.")

    trans_dummies = pd.get_dummies(df['transmission_group'], prefix='trans', drop_first=False).astype(int)
    df = pd.concat([df, trans_dummies], axis=1)

    for col in ['transmission', 'transmission_group']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print(f"-> '{col}' kolonu kaldırıldı.")

    print("\nYeni Transmission Encoding Sütunları (1/0):")
    print(list(trans_dummies.columns))
    return df


# ------------------------------------------------
# 4. Interior & Exterior Color Analizi + Encoding
# ------------------------------------------------
def analyze_color_columns(df):
    print("\n--- Adım: Interior ve Exterior Color Analizi ---")
    for col in ['int_col', 'ext_col']:
        if col not in df.columns:
            print(f"\n⚠️  Uyarı: '{col}' kolonu veri setinde bulunamadı, atlandı.")
            continue
        print(f"\n>>> {col.upper()} Kolonu Analizi:")
        total = len(df)
        missing = df[col].isna().sum()
        print(f"Toplam kayıt sayısı: {total}")
        print(f"Eksik değer sayısı: {missing}")
        print(f"Benzersiz renk sayısı: {df[col].nunique(dropna=True)}")
        print("\nEn sık geçen 10 renk:")
        print(df[col].value_counts().head(10))
    print("\n✅ Interior & Exterior color analizi tamamlandı.\n")
    return df


def process_color_columns_clean(df):
    print("\n--- Adım: Interior ve Exterior Color Temizleme ve Encoding ---")
    invalid_values = ['–', 'nan', '', 'unknown', 'Unknown', 'NONE', 'None', 'NaN']
    for col in ['int_col', 'ext_col']:
        if col not in df.columns:
            print(f"⚠️  '{col}' kolonu bulunamadı, atlandı.")
            continue
        before_len = len(df)
        df = df[~df[col].isin(invalid_values)]
        after_len = len(df)
        print(f"-> {col}: {before_len - after_len} geçersiz kayıt silindi.")
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int)
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)
        print(f"-> '{col}' kolonu encode edildi. ({len(dummies.columns)} sütun)")
    print("\n✅ Interior & Exterior renk kolonları temizlendi ve encode edildi.")
    return df


# ------------------------------------------------
# 5. Price, Engine, Age vb. Ek Fonksiyonlar
# ------------------------------------------------
def clean_price_and_milage(df):
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    df['milage'] = pd.to_numeric(df['milage'].astype(str).str.replace(' mi.', '').str.replace(',', ''), errors='coerce')
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
    def extract_horsepower(s):
        m = re.search(r'(\d+\.?\d*)\s*HP', str(s))
        return float(m.group(1)) if m else np.nan
    def extract_volume(s):
        m = re.search(r'(\d+\.?\d*)\s*(?:L|Liter)', str(s))
        return float(m.group(1)) if m else np.nan
    def extract_cylinders(s):
        if pd.isna(s): return np.nan
        s = str(s)
        if m := re.search(r'(\d+)\s*(?:Cylinder|Cyl)', s): return int(m.group(1))
        if v := re.search(r'V(\d+)', s): return int(v.group(1))
        return np.nan

    df['horsepower'] = df['engine'].apply(extract_horsepower)
    df['engine_volume_L'] = df['engine'].apply(extract_volume)
    df['cylinder_count'] = df['engine'].apply(extract_cylinders)
    if 'fuel_Electric' in df.columns:
        df.loc[df['fuel_Electric'] == 1, ['engine_volume_L', 'cylinder_count']] = 0
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
# 6. Brand & Model Analizi + Encoding
# ------------------------------------------------
def analyze_brand_model(df):
    print("\n--- Adım: Brand ve Model Analizi ---")
    if 'brand' in df.columns:
        print(f"\n>>> BRAND Kolonu Analizi:")
        print(f"Benzersiz marka sayısı: {df['brand'].nunique()}")
        print("\nEn sık geçen 10 marka:")
        print(df['brand'].value_counts().head(10))
        if 'price' in df.columns:
            avg = df.groupby('brand')['price'].mean().sort_values(ascending=False)
            print("\n💎 En pahalı 10 marka:")
            print(avg.head(10))
            print("\n💸 En ucuz 10 marka:")
            print(avg.tail(10))
    if 'model' in df.columns:
        print(f"\n>>> MODEL Kolonu Analizi:")
        print(f"Benzersiz model sayısı: {df['model'].nunique()}")
        print("\nEn sık geçen 10 model:")
        print(df['model'].value_counts().head(10))
    print("\n✅ Brand & Model analizi tamamlandı.\n")
    return df


def encode_brand_model(df):
    print("\n--- Adım: Brand ve Model Encoding ---")
    luxury_brands = ['Bugatti', 'Rolls-Royce', 'Lamborghini', 'Ferrari', 'McLaren',
                     'Maserati', 'Bentley', 'Aston', 'Lucid', 'Rivian']
    economy_brands = ['smart', 'Suzuki', 'Scion', 'Saturn', 'Pontiac',
                      'FIAT', 'Mercury', 'Chrysler']

    df['is_luxury_brand'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)
    df['is_economy_brand'] = df['brand'].apply(lambda x: 1 if x in economy_brands else 0)

    if 'brand' in df.columns:
        top_brands = df['brand'].value_counts().nlargest(20).index
        df['brand_grouped'] = df['brand'].where(df['brand'].isin(top_brands), 'Other')
        brand_dummies = pd.get_dummies(df['brand_grouped'], prefix='brand', drop_first=False).astype(int)
        df = pd.concat([df, brand_dummies], axis=1)
        df.drop(['brand', 'brand_grouped'], axis=1, inplace=True)
        print(f"-> Brand encoding tamamlandı. ({len(brand_dummies.columns)} sütun)")
    if 'model' in df.columns:
        top_models = df['model'].value_counts().nlargest(50).index
        df['model_grouped'] = df['model'].where(df['model'].isin(top_models), 'Other')
        model_dummies = pd.get_dummies(df['model_grouped'], prefix='model', drop_first=False).astype(int)
        df = pd.concat([df, model_dummies], axis=1)
        df.drop(['model', 'model_grouped'], axis=1, inplace=True)
        print(f"-> Model encoding tamamlandı. ({len(model_dummies.columns)} sütun)")
    print("\n✅ Brand & Model encoding işlemi tamamlandı.")
    return df


# ------------------------------------------------
# 7. ANA ÇALIŞMA AKIŞI
# ------------------------------------------------
def main_workflow():
    df = load_data(INPUT_FILE_NAME)
    if df is None:
        return

    df = clean_price_and_milage(df)
    df = prepare_binary_columns(df)
    df = analyze_fuel_type(df)
    df = clean_transmission(df)
    df = analyze_color_columns(df)
    df = process_color_columns_clean(df)
    df = extract_engine_features(df)
    df = analyze_brand_model(df)
    df = encode_brand_model(df)
    df = calculate_car_age(df, CURRENT_YEAR)

    df.to_csv(OUTPUT_FILE_NAME, index=False)
    print(f"\n✅ Tüm işlem tamamlandı! '{OUTPUT_FILE_NAME}' dosyasına kaydedildi.")


if __name__ == "__main__":
    main_workflow()
