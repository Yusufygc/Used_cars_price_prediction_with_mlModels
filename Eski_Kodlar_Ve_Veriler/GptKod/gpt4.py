import pandas as pd
import numpy as np
import re
import os

# Genel Ayarlar
INPUT_FILE_NAME = 'ml/used_cars.csv'
OUTPUT_FILE_NAME = 'ml/GPT4.3.1_cleaned_and_engineered_cars.csv'
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
# 3. TRANSMISSION Temizleme + Encoding
# ------------------------------------------------
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

    df['transmission_group'] = df['transmission'].apply(classify_transmission)

    # Electric araçlar için "Single-Speed" ayarı
    if 'fuel_Electric' in df.columns:
        electric_before = df['fuel_Electric'].sum()
        df.loc[df['fuel_Electric'] == 1, 'transmission_group'] = 'Single-Speed'
        print(f"\n{electric_before} elektrikli araç için transmission_group = 'Single-Speed' olarak ayarlandı.")

    # Unknown verileri sil
    before_len = len(df)
    df = df[df['transmission_group'] != 'Unknown']
    after_len = len(df)
    print(f"{before_len - after_len} satır 'Unknown' transmission nedeniyle silindi.")

    # One-Hot Encoding ve binary dönüşüm
    trans_dummies = pd.get_dummies(df['transmission_group'], prefix='trans', drop_first=False).astype(int)
    df = pd.concat([df, trans_dummies], axis=1)

    # Orijinal kolonu kaldır
    for col in ['transmission', 'transmission_group']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print(f"-> '{col}' kolonu kaldırıldı.")

    print("\nYeni Transmission Encoding Sütunları (1/0):")
    print(list(trans_dummies.columns))
    return df


# ------------------------------------------------
# 4. Interior & Exterior Color Encoding
# ------------------------------------------------

def analyze_color_columns(df):
    """
    'int_col' (interior color) ve 'ext_col' (exterior color) kolonlarını analiz eder.
    - Eksik değer sayılarını ve oranlarını hesaplar
    - En sık geçen ilk 10 rengi listeler
    - Benzersiz renk sayısını gösterir
    - Yazı tiplerini normalize ederek genel dağılımı özetler
    """

    print("\n--- Adım: Interior ve Exterior Color Analizi ---")

    for col in ['int_col', 'ext_col']:
        if col not in df.columns:
            print(f"\n⚠️  Uyarı: '{col}' kolonu veri setinde bulunamadı, atlandı.")
            continue

        print(f"\n>>> {col.upper()} Kolonu Analizi:")
        total = len(df)
        missing = df[col].isna().sum()
        missing_ratio = (missing / total) * 100
        unique_count = df[col].nunique(dropna=True)

        print(f"Toplam kayıt sayısı: {total}")
        print(f"Eksik değer sayısı: {missing} ({missing_ratio:.2f}%)")
        print(f"Benzersiz renk sayısı: {unique_count}")

        # Değer dağılımı (ilk 10)
        print("\nEn sık geçen 10 değer:")
        print(df[col].value_counts(dropna=False).head(10))

        # Normalize edilmiş küçük harfli liste (genel gözlem)
        df[col] = df[col].astype(str).str.lower().str.strip()
        top_renkler = df[col].value_counts().head(10).index.tolist()
        print(f"\nNormalize edilmiş en sık renkler: {top_renkler}")

    print("\n✅ Interior & Exterior color analizi tamamlandı.\n")

    return df

def process_color_columns_clean(df):
    """
    'int_col' ve 'ext_col' kolonlarını işler:
    - Renklerin hepsi korunur (hiçbir sadeleştirme yapılmaz)
    - Geçersiz değerler (–, nan, '', unknown) tamamen silinir
    - Kalan değerler için one-hot encoding uygulanır (binary 1/0)
    - Orijinal kolonlar kaldırılır
    """

    print("\n--- Adım: Interior ve Exterior Color Temizleme ve Encoding ---")

    invalid_values = ['–', 'nan', '', 'unknown', 'Unknown', 'NONE', 'None', 'NaN']

    for col in ['int_col', 'ext_col']:
        if col not in df.columns:
            print(f"⚠️  Uyarı: '{col}' kolonu bulunamadı, atlandı.")
            continue

        # 1️⃣ NaN'leri stringe çevir, boşlukları temizle
        df[col] = df[col].astype(str).str.strip()

        # 2️⃣ Geçersiz değerleri kaldır
        before_len = len(df)
        df = df[~df[col].isin(invalid_values)]
        after_len = len(df)
        print(f"-> {col}: {before_len - after_len} geçersiz kayıt silindi.")

        # 3️⃣ Encoding (renkleri olduğu gibi tutar)
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int)
        df = pd.concat([df, dummies], axis=1)

        # 4️⃣ Orijinal kolonu kaldır
        df.drop(col, axis=1, inplace=True)
        print(f"-> '{col}' kolonu encode edildi ve kaldırıldı. ({len(dummies.columns)} sütun)")

    print("\n✅ Interior & Exterior renk kolonları temizlendi, geçersiz veriler çıkarıldı ve encode edildi.")
    return df

"""
def process_color_columns_simplified(df):
    
    # 'int_col' ve 'ext_col' sütunlarını sadeleştirerek işler:
    # - Renkleri gruplar (örnek: 'Jet Black', 'Charcoal Black' → 'black')
    # - Eksik değerleri 'unknown' olarak doldurur
    # - One-hot encoding (binary 1/0) uygular
    # - Orijinal kolonları kaldırır
    

    print("\n--- Adım: Interior ve Exterior Color Sadeleştirme ---")

    # Renk kökleri
    base_colors = ['black', 'white', 'gray', 'silver', 'blue', 'red', 'green', 'brown', 'beige', 'gold', 'yellow', 'orange']

    def simplify_color(value):
        value = str(value).lower().strip()
        for color in base_colors:
            if color in value:
                return color
        if value in ['–', 'nan', '', 'unknown']:
            return 'unknown'
        return 'other'

    for col in ['int_col', 'ext_col']:
        if col not in df.columns:
            print(f"⚠️  Uyarı: '{col}' kolonu bulunamadı, atlandı.")
            continue

        df[col] = df[col].fillna('unknown').apply(simplify_color)

        # Encoding
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int)
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)

        print(f"-> '{col}' kolonu sadeleştirildi ve encode edildi. ({len(dummies.columns)} sütun)")

    print("\n✅ Interior & Exterior renk kolonları sadeleştirildi ve encode edildi.")
    return df


"""

# ------------------------------------------------
# 5. Diğer Temizlik ve Özellik Fonksiyonları
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
# 6. analyze_brand_model
# ------------------------------------------------
def analyze_brand_model(df):
    """
    'brand' ve 'model' kolonlarını analiz eder.
    - Kaç farklı marka/model olduğunu bulur
    - En sık geçen markaları ve modelleri listeler
    - Ortalama fiyat ile markalar arasında ilişkiyi inceler
    - En pahalı ve en ucuz markaları gösterir
    """

    print("\n--- Adım: Brand ve Model Analizi ---")

    # --- BRAND Analizi ---
    if 'brand' in df.columns:
        print(f"\n>>> BRAND Kolonu Analizi:")
        total = len(df)
        unique_brands = df['brand'].nunique()
        print(f"Toplam araç sayısı: {total}")
        print(f"Benzersiz marka sayısı: {unique_brands}")

        print("\nEn sık geçen 10 marka:")
        print(df['brand'].value_counts().head(10))

        if 'price' in df.columns:
            brand_avg_prices = df.groupby('brand')['price'].mean().sort_values(ascending=False)

            print("\n💎 Fiyat ortalaması en yüksek 10 marka:")
            print(brand_avg_prices.head(10))

            print("\n💸 Fiyat ortalaması en düşük 10 marka:")
            print(brand_avg_prices.tail(10))

            # Fiyat aralıklarını göstermek için min & max tablo
            price_range = df.groupby('brand')['price'].agg(['min', 'max', 'mean', 'count']).sort_values('mean', ascending=False)
            print("\n📊 Marka bazlı fiyat aralıkları (İlk 10):")
            print(price_range.head(10))

    else:
        print("⚠️  'brand' kolonu bulunamadı.")

    # --- MODEL Analizi ---
    if 'model' in df.columns:
        print(f"\n>>> MODEL Kolonu Analizi:")
        unique_models = df['model'].nunique()
        print(f"Benzersiz model sayısı: {unique_models}")

        print("\nEn sık geçen 10 model:")
        print(df['model'].value_counts().head(10))

        if 'brand' in df.columns:
            print("\nModel çeşitliliği en yüksek 5 marka:")
            top_brands_by_models = (
                df.groupby('brand')['model']
                .nunique()
                .sort_values(ascending=False)
                .head(5)
            )
            print(top_brands_by_models)
    else:
        print("⚠️  'model' kolonu bulunamadı.")

    print("\n✅ Brand & Model analizi tamamlandı.\n")
    return df




# ------------------------------------------------
# 6. ANA ÇALIŞMA AKIŞI
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

    print("\n--- Adım 4: Transmission Temizliği ---")
    df = clean_transmission(df)

    print("\n--- Adım 4.7: Interior ve Exterior Colors ---")
    df = analyze_color_columns(df)
    df = process_color_columns_clean(df)

    print("\n--- Adım 5: Engine Özellikleri ---")
    df = extract_engine_features(df)

    print("\n--- Adım 5: MARKA MODELİ ---")
    df = analyze_brand_model(df)

    print("\n--- Adım 6: Araç Yaşı ---")
    df = calculate_car_age(df, CURRENT_YEAR)

    print("\n--- Adım 7: Kaydetme ---")
    df.to_csv(OUTPUT_FILE_NAME, index=False)
    print(f"\n✅ İşlem tamamlandı. '{OUTPUT_FILE_NAME}' dosyasına kaydedildi.")


if __name__ == "__main__":
    main_workflow()
