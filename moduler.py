import pandas as pd
import numpy as np
import re
import os

# Genel Ayarlar
INPUT_FILE_NAME = 'ml/used_cars.csv'
OUTPUT_FILE_NAME = 'ml/cleaned_and_engineered_cars_fueltype_cleaned2.csv'
CURRENT_YEAR = 2025 # Araç yaşını hesaplamak için sabit yıl

# ------------------------------------------------
# 1. Temizlik ve Dönüşüm Fonksiyonları (Modüller)
# ------------------------------------------------

def load_data(file_name):
    """Veri setini yükler ve ilk analizleri yapar."""
    try:
        df = pd.read_csv(file_name)
        print(f"'{file_name}' dosyası başarıyla yüklendi. Satır sayısı: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"HATA: '{file_name}' dosyası bulunamadı.")
        return None

def clean_price_and_milage(df):
    """'price' ve 'milage' sütunlarını temizler ve sayısal tipe dönüştürür."""
    
    # --- Price Temizliği ---
    # Metin tipine dönüştür
    df['price'] = df['price'].astype(str)
    # Dolar işareti, virgül ve tırnak işaretlerini kaldır
    df['price'] = df['price'].str.replace('$', '', regex=False)
    df['price'] = df['price'].str.replace(',', '', regex=False)
    df['price'] = df['price'].str.replace('"', '', regex=False)
    # Sayısal tipe dönüştür, hata olursa NaN yap
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    print("-> 'price' kolonu temizlendi ve sayısal tipe dönüştürüldü.")

    # --- Milage Temizliği ---
    # Metin tipine dönüştür
    df['milage'] = df['milage'].astype(str)
    # " mi." ifadesini ve virgülü kaldır
    df['milage'] = df['milage'].str.replace(' mi.', '', regex=False)
    df['milage'] = df['milage'].str.replace(',', '', regex=False)
    # Sayısal tipe dönüştür, hata olursa NaN yap
    df['milage'] = pd.to_numeric(df['milage'], errors='coerce')
    print("-> 'milage' kolonu temizlendi ve sayısal tipe dönüştürüldü.")

    return df


def prepare_binary_columns(df):
    """'clean_title' ve 'accident' gibi ikili (binary) kategorik sütunları işler."""
    
    # --- clean_title İşlemi ---
    # Boş (NaN) değerleri 0 ile değiştir (varsayım: bilinmeyen başlık temiz değil)
    df['clean_title'].fillna(0, inplace=True)
    # 'Yes' değerlerini 1 ile değiştir
    df['clean_title'] = df['clean_title'].replace('Yes', 1)
    # 0 ve 1 dışındaki kalan boş değerleri 0 yap (varsayım: eğer hala NaN, o da temiz değil)
    df['clean_title'] = pd.to_numeric(df['clean_title'], errors='coerce').fillna(0).astype(int)
    print("-> 'clean_title' kolonu ikili formata dönüştürüldü (Yes: 1, Diğer: 0).")
    
    # --- accident İşlemi ---
    # 'accident_binary' adında yeni bir kolon oluştur. 
    # 'None reported' değerlerini 0 (güvenli) diğerlerini 1 (hasarlı/kaza) yap.
    df['accident_binary'] = np.where(df['accident'] == 'None reported', 0, 1)
    df.drop('accident', axis=1, inplace=True)
    print("-> 'accident' kolonu 'accident_binary' adıyla ikili formata dönüştürüldü ve orijinali kaldırıldı.")

    return df


def extract_engine_features(df):
    """'engine' sütununu HP, Hacim (L) ve Silindir sayısına ayırır ve eksikleri işler."""
    
    # --- Regex Çıkarım Fonksiyonları ---

    def extract_horsepower(engine_str):
        """Beygir gücünü (HP) çıkarır."""
        if pd.isna(engine_str) or str(engine_str) in ('–', 'nan'):
            return np.nan
        match = re.search(r'(\d+\.?\d*)\s*HP', str(engine_str))
        return float(match.group(1)) if match else np.nan

    def extract_engine_volume(engine_str):
        """Motor hacmini (Litre/L) çıkarır."""
        if pd.isna(engine_str) or str(engine_str) in ('–', 'nan'):
            return np.nan
        # Litre veya Liter kelimeleri ile sayı eşleştir
        match = re.search(r'(\d+\.?\d*)\s*(?:L|Liter)', str(engine_str))
        return float(match.group(1)) if match else np.nan

    def extract_cylinder_count(engine_str):
        """Silindir sayısını çıkarır."""
        if pd.isna(engine_str) or str(engine_str) in ('–', 'nan'):
            return np.nan
        
        engine_str = str(engine_str)
        # 1. Pattern: N Cylinder/Cyl (Örn: '4 Cylinder Engine')
        match = re.search(r'(\d+)\s*(?:Cylinder|Cyl|cyl)', engine_str)
        if match: return int(match.group(1))
            
        # 2. Pattern: V-N formatı (Örn: 'V8', 'V-12')
        match_v = re.search(r'V(\d+)', engine_str, re.IGNORECASE)
        if match_v: return int(match_v.group(1))
            
        # 3. Pattern: Düz rakam (Örn: "Straight 6", "Flat 6")
        match_s = re.search(r'(Straight|Flat)\s*(\d+)', engine_str, re.IGNORECASE)
        if match_s: return int(match_s.group(2)) # ikinci grubu (sayıyı) al

        return np.nan

    # --- Uygulama ---
    df['horsepower'] = df['engine'].apply(extract_horsepower)
    df['engine_volume_L'] = df['engine'].apply(extract_engine_volume)
    df['cylinder_count'] = df['engine'].apply(extract_cylinder_count)

    # --- Eksik/Elektrikli Değerleri İşleme (Imputation) ---

    # Elektrikli/Hibrit (Plug-In Electric) araçları tespit etme
    electric_terms = ['Electric', 'Plug-In Electric']
    is_electric = df['fuel_type'].astype(str).str.contains('|'.join(electric_terms), case=False, na=False)

    # Elektrikli Araçlar İçin:
    df.loc[is_electric, 'engine_volume_L'] = df.loc[is_electric, 'engine_volume_L'].fillna(0.0)
    df.loc[is_electric, 'cylinder_count'] = df.loc[is_electric, 'cylinder_count'].fillna(0)
    
    # Kalan Eksik Değerleri Medyan ile Doldurma (Güç ve Hacim)
    df['horsepower'].fillna(df['horsepower'].median(), inplace=True)
    df['engine_volume_L'].fillna(df['engine_volume_L'].median(), inplace=True)

    # Kalan Silindir Eksiklerini Medyan ile Doldurma ve Tamsayıya Çevirme
    df['cylinder_count'].fillna(df['cylinder_count'].median(), inplace=True)
    df['cylinder_count'] = df['cylinder_count'].astype(int)

    # Orijinal sütunu kaldır
    df.drop('engine', axis=1, inplace=True)
    print("-> 'engine' kolonu 3 sayısal özelliğe ayrıldı ve orijinali kaldırıldı. (HP, Litre, Silindir Sayısı)")

    return df

def calculate_car_age(df, current_year):
    """'model_year' sütunundan araç yaşını hesaplar ve orijinal sütunu kaldırır."""
    df['car_age'] = current_year - df['model_year']
    df.drop('model_year', axis=1, inplace=True)
    print("-> 'model_year' kullanılarak 'car_age' (araç yaşı) hesaplandı ve orijinali kaldırıldı.")
    return df

def analyze_fuel_type(df):
    """
    'fuel_type' kolonunu analiz eder, sınıflandırır ve temiz bir formata dönüştürür.
    
    İşlevleri:
    - Mevcut yakıt türü dağılımını ekrana yazar.
    - Eksik değerleri 'Unknown' ile doldurur.
    - Benzer yakıt türlerini tek bir kategori altında birleştirir.
    - Yeni bir 'fuel_group' kolonu oluşturur.
    - 'Other' kategorisindeki verileri kaldırır.
    - 'Unknown' kategorisindeki verileri 'Electric' olarak yeniden sınıflandırır.
    """

    print("\n--- Adım: Fuel Type Analizi ---")

    # 1️⃣ Yakıt türü dağılımını göster
    print("\nYakıt Türü Dağılımı:")
    print(df['fuel_type'].value_counts(dropna=False))
    print(f"\nEksik değer sayısı: {df['fuel_type'].isna().sum()}")

    # 2️⃣ Eksik değerleri doldur
    df['fuel_type'] = df['fuel_type'].fillna('Unknown')

    # 3️⃣ Yakıt türlerini sınıflandır
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

    # 4️⃣ Yeni grup kolonunu oluştur
    df['fuel_group'] = df['fuel_type'].apply(classify_fuel_type)

    # 5️⃣ İstenmeyen kategorileri işle
    before_len = len(df)

    # 'Unknown' → 'Electric'
    df['fuel_group'] = df['fuel_group'].replace('Unknown', 'Electric')

    # 'Other' kategorisini kaldır
    df = df[df['fuel_group'] != 'Other']

    after_len = len(df)

    print(f"\n{before_len - after_len} satır 'Other' kategorisinden silindi.")
    print("Tüm 'Unknown' değerler 'Electric' kategorisine eklendi.")

    # 6️⃣ Sonuç dağılımını göster
    print("\nYeni Gruplandırılmış Yakıt Türü Dağılımı (fuel_group):")
    print(df['fuel_group'].value_counts())

    return df

def analyze_transmission(df):
    """
    'transmission' (vites tipi) kolonunu analiz eder.
    
    Bu fonksiyon sadece analiz yapar, veri üzerinde değişiklik yapmaz.
    - Kategorilerin dağılımını gösterir.
    - Eksik değer sayılarını ve oranlarını hesaplar.
    - Benzersiz değerleri listeler.
    """

    print("\n--- Adım: Transmission Kolonu Analizi ---")

    # 1️⃣ Kolon var mı kontrol et
    if 'transmission' not in df.columns:
        print("HATA: 'transmission' kolonu veri setinde bulunamadı.")
        return df

    # 2️⃣ Eksik değer analizi
    total = len(df)
    missing = df['transmission'].isna().sum()
    missing_ratio = (missing / total) * 100

    print(f"\nToplam kayıt sayısı: {total}")
    print(f"Eksik değer sayısı: {missing} ({missing_ratio:.2f}%)")

    # 3️⃣ Değer dağılımı
    print("\nTransmission Tipi Dağılımı:")
    print(df['transmission'].value_counts(dropna=False))

    # 4️⃣ Benzersiz değerleri listele
    unique_values = df['transmission'].dropna().unique()
    print(f"\nBenzersiz değerler ({len(unique_values)} adet):")
    print(unique_values)

    # 5️⃣ Örnek veri tipini göster
    print(f"\nVeri tipi: {df['transmission'].dtype}")
    print("Bu kolonda genellikle 'Automatic', 'Manual', 'CVT', 'Semi-Auto' gibi değerler bulunur.")

    return df

def clean_transmission(df):
    """
    'transmission' kolonunu temizler, gruplaştırır ve modellemeye hazır hale getirir.
    
    - Farklı yazım biçimlerini ortak kategorilere çevirir.
    - Gereksiz veya hatalı değerleri kaldırır.
    - Yeni bir 'transmission_group' kolonu oluşturur.
    - Unknown değerleri siler.
    - Electric araçların vites tipini 'Single-Speed' olarak günceller.
    """

    print("\n--- Adım: Transmission Kolonu Temizliği ---")

    if 'transmission' not in df.columns:
        print("HATA: 'transmission' kolonu veri setinde bulunamadı.")
        return df

    def classify_transmission(t):
        if pd.isna(t):
            return 'Unknown'
        t = str(t).lower().strip()

        # 1️⃣ Otomatik
        if any(x in t for x in ['a/t', 'automatic', 'auto,', 'at', 'aut']):
            return 'Automatic'

        # 2️⃣ Manuel
        elif any(x in t for x in ['m/t', 'manual', 'mt']):
            return 'Manual'

        # 3️⃣ CVT (Sürekli değişken)
        elif 'cvt' in t:
            return 'CVT'

        # 4️⃣ Dual Clutch veya Auto-Shift
        elif any(x in t for x in ['dual', 'dct', 'auto-shift', 'shift mode']):
            return 'Semi-Automatic'

        # 5️⃣ Tek vitesli (Elektrikli)
        elif any(x in t for x in ['single-speed', '1-speed', '2-speed']):
            return 'Single-Speed'

        # 6️⃣ Geçersiz değerler
        elif any(x in t for x in ['variable', '–', 'scheduled', 'overdrive', 'f']):
            return 'Unknown'

        else:
            return 'Unknown'

    # Yeni grup kolonu oluştur
    df['transmission_group'] = df['transmission'].apply(classify_transmission)

    # Electric araçlara özel kural
    if 'fuel_group' in df.columns:
        electric_before = (df['fuel_group'] == 'Electric').sum()
        df.loc[df['fuel_group'] == 'Electric', 'transmission_group'] = 'Single-Speed'
        print(f"\n{electric_before} elektrikli araç için transmission_group = 'Single-Speed' olarak ayarlandı.")

    # Unknown verileri sil
    before_len = len(df)
    df = df[df['transmission_group'] != 'Unknown']
    after_len = len(df)
    print(f"{before_len - after_len} satır 'Unknown' transmission nedeniyle silindi.")

    # Dağılımı göster
    print("\nGüncellenmiş Transmission Grupları Dağılımı:")
    print(df['transmission_group'].value_counts())

    return df



# ------------------------------------------------
# 2. Ana Çalışma Akışı
# ------------------------------------------------

def main_workflow():
    """Tüm veri temizleme ve mühendislik adımlarını sırasıyla uygular."""
    
    # 1. Veri Yükleme
    df = load_data(INPUT_FILE_NAME)
    if df is None:
        return

    # İlk Analiz ve Raporlama
    print("\n--- İlk Veri Analizi ---")
    print(df.head())
    print(df.info())
    print("\n'clean_title' Değer Dağılımı (Ham):")
    print(df['clean_title'].value_counts(dropna=False))
    print("\n'accident' Değer Dağılımı (Ham):")
    print(df['accident'].value_counts(dropna=False))

    # 2. Temel Veri Temizliği ve Format Dönüşümü
    print("\n--- Adım 1: Temel Veri Temizliği (Price ve Milage) ---")
    df = clean_price_and_milage(df)

    # 3. İkili (Binary) Sütunların Hazırlanması
    print("\n--- Adım 2: İkili Kategorik Sütunların Hazırlanması (clean_title ve accident) ---")
    df = prepare_binary_columns(df)

    # 3.5. Fuel Type Analizi ve Gruplandırma
    print("\n--- Adım 3.5: Fuel Type Analizi ve Gruplandırma ---")
    df = analyze_fuel_type(df)

    # 4. Engine Sütunundan Özellik Mühendisliği (Regresyon için kritik)
    print("\n--- Adım 3: Engine Özellik Mühendisliği ---")
    df = extract_engine_features(df)

    # 4.5. Transmission Analizi
    print("\n--- Adım 4.5: Transmission Analizi ---")
    df = analyze_transmission(df)
    df = clean_transmission(df)


    # 5. Araç Yaşı Hesaplama
    print("\n--- Adım 5: Araç Yaşını Hesaplama ---")
    df = calculate_car_age(df, CURRENT_YEAR)

    # 6. Kontrol ve Kaydetme
    print("\n--- SONUÇ KONTROLÜ ve Kayıt ---")
    print("\nGüncellenmiş Veri Seti (İlk 5 Satır):")
    print(df.head())
    print("\nGüncellenmiş Sütun Bilgileri (Veri Tipleri):")
    print(df.info())

    # Sonuç veri setini kaydet
    df.to_csv(OUTPUT_FILE_NAME, index=False)
    print(f"\nİşlem tamamlandı. Temizlenmiş ve mühendislik uygulanmış veri seti '{OUTPUT_FILE_NAME}' dosyasına kaydedildi.")

# Ana fonksiyonu çalıştır
main_workflow()