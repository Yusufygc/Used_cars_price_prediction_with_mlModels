"""Ham araç kaydından fiyat tahmini (aralıklı).

Kullanım:
    python -m src.predict --json '{"brand": "Ford", "model": "F-150 XLT", "model_year": 2019,
        "milage": "45,000 mi.", "fuel_type": "Gasoline", "engine": "375.0HP 5.0L V8",
        "transmission": "10-Speed A/T", "ext_col": "Black", "int_col": "Gray",
        "accident": "None reported", "clean_title": "Yes"}'
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .config import BUNDLE_PATH
from .features import parse_raw

RAW_COLUMNS = ["brand", "model", "model_year", "milage", "fuel_type", "engine", "transmission",
               "ext_col", "int_col", "accident", "clean_title"]


def predict_raw(records: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    for c in RAW_COLUMNS:
        if c not in records:
            records = records.assign(**{c: np.nan})
    X = parse_raw(records, bundle["reference_year"])[bundle["features"]]
    pred = bundle["model"].predict(X)
    q = bundle["interval_log_q90"]
    low_90 = np.expm1(np.log1p(pred) - q)
    high_90 = np.expm1(np.log1p(pred) + q)
    return pd.DataFrame({"prediction": pred, "low_90": low_90, "high_90": high_90},
                        index=records.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="tek araç, ham şema (used_cars.csv sütunları)")
    args = ap.parse_args()
    bundle = joblib.load(BUNDLE_PATH)
    out = predict_raw(pd.DataFrame([json.loads(args.json)]), bundle).iloc[0]
    print(f"Tahmin: ${out.prediction:,.0f}  (%90 aralık: ${out.low_90:,.0f} - ${out.high_90:,.0f})")


if __name__ == "__main__":
    main()
