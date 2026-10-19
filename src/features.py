"""Ham veriyi modele uygun tablolara çeviren, durumsuz (stateless) ayrıştırıcılar.

Burada hiçbir istatistik öğrenilmez (medyan, ortalama yok). Öğrenme gerektiren her şey
pipeline.py içinde Pipeline'a konur, böylece data leakage olmaz.
"""
import re

import numpy as np
import pandas as pd

from .config import COLORS, EV_BRANDS

_MISSING_TOKENS = {"", "-", "–", "—", "n/a", "na", "nan", "none", "null", "not supported", "belirtilmemiş"}
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_TR_THOUSAND_DOT = re.compile(r"(?<=\d)\.(?=\d{3}(?!\d))")


def _is_missing(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in _MISSING_TOKENS


def _to_number(value) -> float:
    if _is_missing(value):
        return np.nan
    s = str(value)
    if "," in s:
        cleaned = s.replace(",", "")
    else:
        cleaned = _TR_THOUSAND_DOT.sub("", s)
    m = _NUM_RE.search(cleaned)
    return float(m.group(0)) if m else np.nan


def parse_price(value) -> float:
    """'$10,300' -> 10300.0 | '1.515.000 TL' -> 1515000.0"""
    return _to_number(value)


def parse_mileage(value) -> float:
    """'51,000 mi.' -> 51000.0 | '217.000 KM' -> 217000.0"""
    return _to_number(value)


def classify_fuel(brand, fuel_type, engine) -> str:
    """Tek yakıt sınıflandırma fonksiyonu (TR & EN uyumlu)."""
    b = "" if _is_missing(brand) else str(brand).strip()
    fuel = "" if _is_missing(fuel_type) else str(fuel_type).strip().lower()
    eng = "" if _is_missing(engine) else str(engine).strip().lower()
    if b in EV_BRANDS or b.upper() in EV_BRANDS:
        return "electric"
    if "hybrid" in fuel or "hibrit" in fuel or "hybrid" in eng or "hibrit" in eng or "plug-in" in fuel:
        return "hybrid"
    if ("electric" in fuel or "elektrik" in fuel or "electric" in eng or "elektrik" in eng
            or "battery" in eng or re.search(r"\b\d+\.?\d*\s*(?:ah|kwh|kw)\b", eng)):
        return "electric"
    if "lpg" in fuel:
        return "lpg"
    if "diesel" in fuel or "dizel" in fuel or "diesel" in eng or "dci" in eng or "tdi" in eng or "multijet" in eng or "bluehdi" in eng:
        return "diesel"
    if "e85" in fuel or "flex" in fuel:
        return "flex_fuel"
    if "gas" in fuel or "petrol" in fuel or "benzin" in fuel or "gasoline" in eng or "unleaded" in eng:
        return "gasoline"
    return "unknown" if not fuel else "other"


def classify_transmission(text, fuel_category) -> str:
    """Sıra önemli: tek vites/elektrikli, CVT, yarı otomatik, otomatik, manuel (TR & EN uyumlu)."""
    t = "" if _is_missing(text) else str(text).upper()
    if re.search(r"SINGLE[- ]SPEED|\b1-SPEED\b|FIXED", t) or fuel_category == "electric":
        return "single_speed"
    if re.search(r"\bCVT\b|CONTINUOUSLY VARIABLE|\bVARIABLE\b", t):
        return "cvt"
    if re.search(r"DUAL[- ]SHIFT|\bDCT\b|\bPDK\b|\bDSG\b|SEMI|YARI\s*OTOMAT", t):
        return "semi_auto"
    if re.search(r"\bA/?T\b|\bAUTO(?:MATIC)?\b|OVERDRIVE|OTOMAT", t):
        return "automatic"
    if re.search(r"\bM/T\b|\bMT\b|MANUAL|MANUEL|\bDÜZ\b|\bDUZ\b", t):
        return "manual"
    return "other"


_HP_RANGE = re.compile(r"(\d+)\s*-\s*(\d+)\s*(?:HP|BG|PS)\b", re.I)
_HP = re.compile(r"(\d+\.?\d*)\s*(?:HP|BG|PS)\b", re.I)
_LITER = re.compile(r"(\d+\.?\d*)\s*(?:L|LITER)\b", re.I)
_CC_RANGE = re.compile(r"(\d{3,4})\s*-\s*(\d{3,4})\s*(?:CC|CM3)\b", re.I)
_CC = re.compile(r"(\d{3,4})\s*(?:CC|CM3)\b", re.I)
_DISP_TOKEN = re.compile(r"\b([0-6]\.\d)\s*(?:DCI|TDI|TSI|TFSI|TCE|MPI|FIRE|MULTIJET|MJET|BLUEHDI|HDI|PURETECH|VTEC|I-VTEC|CDI|CGI|GDI|CRDI|ECOBOOST|TDCI|CDTI|VTI|THP)\b", re.I)
_CYL_LAYOUT = re.compile(r"\b(?:[VIHW]-?|STRAIGHT\s+|FLAT\s+)(\d{1,2})\b", re.I)
_CYL_WORD = re.compile(r"(\d+)\s*(?:CYLINDER|SİLİNDİR|SILINDIR)", re.I)


def parse_engine(text, fuel_category) -> tuple:
    """'300.0HP 3.7L V6 ...' veya '90 HP 1461 CC 1.5 dCi' -> (hp, litre, silindir)."""
    t = "" if _is_missing(text) else str(text)
    hp_r = _HP_RANGE.search(t)
    if hp_r:
        hp = (float(hp_r.group(1)) + float(hp_r.group(2))) / 2.0
    else:
        hp_m = _HP.search(t)
        hp = float(hp_m.group(1)) if hp_m else np.nan

    if fuel_category == "electric":
        return hp, 0.0, 0.0

    liters_m = _LITER.search(t)
    if liters_m:
        liters = float(liters_m.group(1))
    else:
        cc_r = _CC_RANGE.search(t)
        if cc_r:
            liters = round(((float(cc_r.group(1)) + float(cc_r.group(2))) / 2.0) / 1000.0, 2)
        else:
            cc_m = _CC.search(t)
            if cc_m:
                liters = round(float(cc_m.group(1)) / 1000.0, 2)
            else:
                disp_m = _DISP_TOKEN.search(t)
                liters = float(disp_m.group(1)) if disp_m else np.nan

    cyl = _CYL_LAYOUT.search(t) or _CYL_WORD.search(t)
    if cyl:
        cyl_val = float(cyl.group(1))
    elif not np.isnan(liters) and (_CC.search(t) or _CC_RANGE.search(t)):
        if liters <= 1.05:
            cyl_val = 3.0
        elif liters <= 2.25:
            cyl_val = 4.0
        elif liters <= 3.6:
            cyl_val = 6.0
        else:
            cyl_val = 8.0
    else:
        cyl_val = np.nan
    return hp, liters, cyl_val


def classify_accident(value) -> str:
    if _is_missing(value):
        return "unknown"
    v = str(value).lower()
    if "none reported" in v or "hasarsız" in v or "tramer yok" in v:
        return "none"
    return "reported"


def classify_clean_title(value) -> str:
    return "missing" if _is_missing(value) else str(value).strip().lower()


_TR_COLOR_MAP = {
    "siyah": "black",
    "beyaz": "white",
    "gri": "gray",
    "füme": "gray",
    "fume": "gray",
    "antrasit": "gray",
    "gümüş": "silver",
    "gumus": "silver",
    "mavi": "blue",
    "lacivert": "blue",
    "turkuaz": "blue",
    "kırmızı": "red",
    "kirmizi": "red",
    "bordo": "red",
    "kahverengi": "brown",
    "bronz": "brown",
    "yeşil": "green",
    "yesil": "green",
    "bej": "beige",
    "şampanya": "beige",
    "krem": "beige",
    "altın": "gold",
    "sar": "yellow",
    "turuncu": "orange",
    "sahibinden": "sahibinden",
    "galeriden": "galeriden",
    "yetkili": "yetkili_bayi",
}


def simplify_color(value) -> str:
    if _is_missing(value):
        return "unknown"
    v = str(value).lower()
    for tr_key, mapped in _TR_COLOR_MAP.items():
        if tr_key in v:
            return mapped
    for c in COLORS:
        if c in v:
            return c
    return "other"


_SPLIT_BRANDS = {
    "Land": ("Rover", "Land Rover"),
    "Aston": ("Martin", "Aston Martin"),
    "Alfa": ("Romeo", "Alfa Romeo"),
}


def _normalize_brand_model(brand, model) -> tuple[str, str]:
    b = "unknown" if _is_missing(brand) else str(brand).strip()
    m = "unknown" if _is_missing(model) else str(model).strip()
    if b in _SPLIT_BRANDS:
        prefix, full_brand = _SPLIT_BRANDS[b]
        b = full_brand
        if m.lower().startswith(prefix.lower()):
            m = m[len(prefix):].strip() or "unknown"
    return b, m


def parse_raw(df: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    """Ham used_cars.csv (veya aynı şemalı tek satır) -> model girdisi.

    'price' sütunu varsa sayıya çevrilir; yoksa atlanır (inference).
    """
    out = pd.DataFrame(index=df.index)
    bm = [_normalize_brand_model(b, m) for b, m in zip(df["brand"], df["model"])]
    out[["brand", "model"]] = pd.DataFrame(bm, index=df.index)
    out["age"] = (reference_year - pd.to_numeric(df["model_year"], errors="coerce")).clip(lower=0)
    out["mileage"] = df["milage"].map(parse_mileage)
    out["fuel_category"] = [classify_fuel(b, f, e) for b, f, e in zip(out["brand"], df["fuel_type"], df["engine"])]
    out["transmission_type"] = [classify_transmission(t, f) for t, f in zip(df["transmission"], out["fuel_category"])]
    engine = [parse_engine(e, f) for e, f in zip(df["engine"], out["fuel_category"])]
    out[["horsepower", "engine_volume", "cylinders"]] = pd.DataFrame(engine, index=df.index)
    out["accident"] = df["accident"].map(classify_accident)
    out["clean_title"] = df["clean_title"].map(classify_clean_title)
    out["ext_col"] = df["ext_col"].map(simplify_color)
    out["int_col"] = df["int_col"].map(simplify_color)
    if "price" in df:
        out["price"] = df["price"].map(parse_price)
    return out
