"""Türkiye ikinci el araç veri toplayıcı (arabam.com).

Popüler otomobil ve SUV marka/kategorilerinden güncel ilanları çeker,
sayfa içindeki yapısal JSON (`var model`) verisini ayrıştırır ve
`data/tr_used_cars.csv` dosyasına kaydeder.

Çalıştırma:
    python -m src.scraper_tr --pages 3
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
from curl_cffi import requests

from .config import ROOT

TR_DATA_PATH = ROOT / "data" / "tr_used_cars.csv"

# Türkiye piyasasında en yüksek hacimli Otomobil ve SUV/Pick-up marka yolları
CATEGORY_PATHS = [
    # Otomobil
    "otomobil/renault",
    "otomobil/fiat",
    "otomobil/volkswagen",
    "otomobil/ford",
    "otomobil/toyota",
    "otomobil/hyundai",
    "otomobil/opel",
    "otomobil/honda",
    "otomobil/peugeot",
    "otomobil/bmw",
    "otomobil/mercedes-benz",
    "otomobil/audi",
    "otomobil/skoda",
    "otomobil/citroen",
    "otomobil/seat",
    "otomobil/kia",
    "otomobil/nissan",
    "otomobil/dacia",
    "otomobil/volvo",
    "otomobil/chevrolet",
    # Arazi, SUV, Pick-up & Elektrikli
    "arazi-suv-pick-up/nissan",
    "arazi-suv-pick-up/dacia",
    "arazi-suv-pick-up/peugeot",
    "arazi-suv-pick-up/volkswagen",
    "arazi-suv-pick-up/hyundai",
    "arazi-suv-pick-up/kia",
    "arazi-suv-pick-up/toyota",
    "arazi-suv-pick-up/ford",
    "arazi-suv-pick-up/bmw",
    "arazi-suv-pick-up/mercedes-benz",
    "arazi-suv-pick-up/land-rover",
    "arazi-suv-pick-up/volvo",
    "arazi-suv-pick-up/chery",
    "arazi-suv-pick-up/togg",
    "arazi-suv-pick-up/tesla",
    "arazi-suv-pick-up/jeep",
    "arazi-suv-pick-up/audi",
    "arazi-suv-pick-up/renault",
    "arazi-suv-pick-up/skoda",
    "arazi-suv-pick-up/opel",
    "arazi-suv-pick-up/fiat",
    "arazi-suv-pick-up/citroen",
    "arazi-suv-pick-up/honda",
    "arazi-suv-pick-up/mg",
    "arazi-suv-pick-up/cupra",
    "arazi-suv-pick-up/byd",
]


def _extract_doc(doc: dict) -> dict | None:
    """Tek bir arabam.com Document JSON kaydını tablo satırına dönüştürür."""
    ad_id = doc.get("Id")
    year = doc.get("Year")
    price_str = doc.get("FormattedPrice")
    km_str = doc.get("Km")
    if not ad_id or not year or not price_str:
        return None

    cats = [c.get("Name", "").strip() for c in doc.get("Categories", []) if c.get("Name")]
    if len(cats) < 3:
        return None

    category = cats[0]
    brand = cats[1]
    series = cats[2]
    engine_trim = cats[3] if len(cats) > 3 else ""
    trim = " ".join(cats[4:]) if len(cats) > 4 else ""
    full_model = " ".join(cats[2:]).strip()

    props = {p.get("Key"): p.get("Value", "") for p in doc.get("DetailedProperties", []) if p.get("Key")}
    prop_list = {p.get("Name"): p.get("Description", "") for p in doc.get("PropertyListWithKey", []) if p.get("Name")}

    fuel_type = props.get("Yakıt tipi", "")
    transmission = props.get("Vites tipi", "")
    motor_hacmi = props.get("Motor hacmi", "")
    motor_gucu = props.get("Motor gücü", "")
    drivetrain = props.get("Çekiş", "")
    body_type = props.get("Kasa tipi", "") or ("SUV" if "SUV" in category else "Otomobil")
    seller_type = props.get("İlan Sahibi", "")
    warranty = props.get("Garanti durumu", "")
    ext_col = prop_list.get("Renk", "")

    engine_parts = [p for p in [motor_gucu, motor_hacmi, engine_trim] if p]
    engine_str = " ".join(engine_parts)

    owned = doc.get("OwnedProperties") or {}
    has_damage = bool(owned.get("HasDamageEntrance"))
    has_tramer = bool(owned.get("HasTramer"))
    accident = "At least 1 accident or damage reported" if (has_damage or has_tramer) else "None reported"
    clean_title = "Yes" if not has_damage else ""

    return {
        "id": ad_id,
        "category": category,
        "brand": brand,
        "series": series,
        "engine_trim": engine_trim,
        "trim": trim,
        "model": full_model,
        "model_year": int(year),
        "milage": km_str or props.get("Kilometre", ""),
        "fuel_type": fuel_type,
        "engine": engine_str,
        "transmission": transmission,
        "drivetrain": drivetrain,
        "body_type": body_type,
        "ext_col": ext_col,
        "int_col": seller_type,  # İlan sahibi (Sahibinden/Galeriden/Yetkili Bayiden)
        "seller_type": seller_type,
        "warranty": warranty,
        "accident": accident,
        "clean_title": clean_title,
        "city": doc.get("City", ""),
        "price": price_str,
    }


def fetch_page(session: requests.Session, path: str, page: int) -> list[dict]:
    url = f"https://www.arabam.com/ikinci-el/{path}?take=50&page={page}"
    resp = session.get(url, impersonate="chrome120", timeout=20)
    if resp.status_code != 200:
        return []
    marker = "var model = "
    idx = resp.text.find(marker)
    if idx == -1:
        return []
    try:
        data, _ = json.JSONDecoder().raw_decode(resp.text[idx + len(marker):])
    except Exception:
        return []
    docs = data.get("Documents") or []
    rows = []
    for d in docs:
        parsed = _extract_doc(d)
        if parsed is not None:
            rows.append(parsed)
    return rows


def scrape_all(pages_per_path: int = 3, delay: float = 0.25, output_path: Path = TR_DATA_PATH) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids = set()
    all_rows = []

    with requests.Session() as session:
        for i, path in enumerate(CATEGORY_PATHS, 1):
            path_count = 0
            for page in range(1, pages_per_path + 1):
                try:
                    rows = fetch_page(session, path, page)
                except Exception as exc:
                    print(f"  [Uyarı] {path} sayfa {page} çekilemedi: {exc}")
                    break
                if not rows:
                    break
                new_rows = [r for r in rows if r["id"] not in seen_ids]
                for r in new_rows:
                    seen_ids.add(r["id"])
                all_rows.extend(new_rows)
                path_count += len(new_rows)
                if len(rows) < 40:
                    break
                time.sleep(delay)
            print(f"[{i:02d}/{len(CATEGORY_PATHS)}] {path:<32} -> +{path_count:3d} ilan (Toplam: {len(all_rows)})")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nTamamlandı! {len(df)} tekil Türkiye araç ilanı kaydedildi: {output_path}")
    return df


def main():
    ap = argparse.ArgumentParser(description="arabam.com ikinci el araç veri toplayıcı")
    ap.add_argument("--pages", type=int, default=3, help="Her marka/kategori için çekilecek sayfa sayısı (sayfa başı 50 ilan)")
    ap.add_argument("--delay", type=float, default=0.25, help="İstekler arası bekleme süresi (saniye)")
    args = ap.parse_args()
    scrape_all(pages_per_path=args.pages, delay=args.delay)


if __name__ == "__main__":
    main()
