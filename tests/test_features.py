import numpy as np
import pandas as pd
import pytest

from src.features import (classify_accident, classify_clean_title, classify_fuel,
                          classify_transmission, parse_engine, parse_mileage,
                          parse_price, parse_raw, simplify_color)


def test_price_mileage():
    assert parse_price("$10,300") == 10300
    assert parse_mileage("51,000 mi.") == 51000
    assert parse_mileage("51,000.5 mi.") == 51000.5
    assert np.isnan(parse_price(np.nan))
    assert np.isnan(parse_mileage("N/A"))
    assert np.isnan(parse_mileage("N/A."))
    assert np.isnan(parse_price("..."))
    assert np.isnan(parse_mileage("–"))


@pytest.mark.parametrize("brand,fuel,engine,expected", [
    ("Tesla", np.nan, "Dual Motor", "electric"),
    ("Toyota", "Hybrid", "2.5L", "hybrid"),
    ("Ford", "Plug-In Hybrid", "x", "hybrid"),
    ("BMW", "Diesel", "x", "diesel"),
    ("Ford", "E85 Flex Fuel", "x", "flex_fuel"),
    ("Ford", "Gasoline", "x", "gasoline"),
    ("Ford", np.nan, "3.5L V6", "unknown"),
    ("Ford", "–", "–", "unknown"),
    ("Ford", "not supported", "–", "unknown"),
    ("Audi", np.nan, "Electric Motor", "electric"),
    ("Kia", np.nan, "111.2Ah / FR 70kW / RR 160kW (697V)", "electric"),
])
def test_fuel(brand, fuel, engine, expected):
    assert classify_fuel(brand, fuel, engine) == expected


@pytest.mark.parametrize("text,fuel,expected", [
    ("6-Speed A/T", "gasoline", "automatic"),
    ("8-Speed Automatic", "gasoline", "automatic"),
    ("8-SPEED AT", "gasoline", "automatic"),
    ("6-Speed M/T", "gasoline", "manual"),
    ("6-Speed Manual", "gasoline", "manual"),
    ("8-Speed Automatic with Auto-Shift Manual", "gasoline", "automatic"),
    ("CVT Transmission", "gasoline", "cvt"),
    ("7-Speed Dual Shift Mode", "gasoline", "semi_auto"),
    ("Transmission w/Dual Shift Mode", "gasoline", "semi_auto"),
    ("Single-Speed Fixed Gear", "electric", "single_speed"),
    ("1-Speed A/T", "electric", "single_speed"),
    ("A/T", "electric", "single_speed"),
    ("–", "electric", "single_speed"),
    ("–", "gasoline", "other"),
])
def test_transmission(text, fuel, expected):
    assert classify_transmission(text, fuel) == expected


def test_engine():
    assert parse_engine("300.0HP 3.7L V6 Cylinder Engine Flex Fuel Capability", "flex_fuel") == (300.0, 3.7, 6.0)
    hp, liters, cyl = parse_engine("3.8L V6 24V GDI DOHC", "gasoline")
    assert np.isnan(hp) and liters == 3.8 and cyl == 6.0
    hp, liters, cyl = parse_engine("Electric Motor", "electric")
    assert np.isnan(hp) and liters == 0 and cyl == 0
    assert parse_engine("2.0L I4", "gasoline")[2] == 4.0
    assert parse_engine("5.2L V-10 gasoline direct injection", "gasoline")[2] == 10.0
    assert parse_engine("4L V-8 gasoline direct injection", "gasoline")[2] == 8.0
    assert parse_engine("2L I-4 DOHC", "gasoline")[2] == 4.0
    assert parse_engine("382.0HP 3.0L Straight 6 Cylinder Engine", "gasoline")[2] == 6.0
    assert parse_engine("5 Cylinder Engine", "gasoline")[2] == 5.0


def test_categoricals():
    assert classify_accident(np.nan) == "unknown"
    assert classify_accident("–") == "unknown"
    assert classify_accident("None reported") == "none"
    assert classify_accident("At least 1 accident or damage reported") == "reported"
    assert classify_clean_title(np.nan) == "missing"
    assert classify_clean_title("–") == "missing"
    assert classify_clean_title("Yes") == "yes"
    assert simplify_color("Moonlight Cloud Blue") == "blue"
    assert simplify_color("Pewter") == "other"
    assert simplify_color(np.nan) == "unknown"
    assert simplify_color("–") == "unknown"


def test_parse_raw_without_price_column():
    row = pd.DataFrame([
        {
            "brand": "Ford", "model": "F-150", "model_year": 2020, "milage": "10,000 mi.",
            "fuel_type": "Gasoline", "engine": "400.0HP 5.0L V8", "transmission": "10-Speed A/T",
            "ext_col": "Black", "int_col": "Gray", "accident": "None reported", "clean_title": "Yes",
        },
        {
            "brand": "Land", "model": "Rover Range Rover Sport", "model_year": 2028, "milage": "5,000 mi.",
            "fuel_type": "Gasoline", "engine": "3.0L V-6", "transmission": "8-Speed A/T",
            "ext_col": "White", "int_col": "–", "accident": np.nan, "clean_title": np.nan,
        },
    ])
    out = parse_raw(row, reference_year=2026)
    assert "price" not in out
    assert out.loc[0, "age"] == 6 and out.loc[0, "mileage"] == 10000
    assert out.loc[0, "cylinders"] == 8
    assert out.loc[1, "brand"] == "Land Rover" and out.loc[1, "model"] == "Range Rover Sport"
    assert out.loc[1, "age"] == 0  # gelecek model yılı negatif olamaz
    assert out.loc[1, "cylinders"] == 6
    assert out.loc[1, "int_col"] == "unknown"


def test_pipeline_and_predict_and_shap():
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor
    from src.config import FEATURES
    from src.explain import compute_shap
    from src.pipeline import build_model
    from src.predict import predict_raw

    rng = np.random.default_rng(42)
    n = 60
    raw = pd.DataFrame({
        "brand": rng.choice(["Ford", "Toyota", "Tesla"], n),
        "model": rng.choice(["F-150", "Camry", "Model 3"], n),
        "model_year": rng.integers(2015, 2025, n),
        "milage": [f"{m:,} mi." for m in rng.integers(5000, 120000, n)],
        "fuel_type": rng.choice(["Gasoline", "Hybrid", np.nan], n),
        "engine": rng.choice(["300.0HP 3.5L V-6", "2.5L I-4", "Electric Motor"], n),
        "transmission": rng.choice(["8-Speed A/T", "1-Speed A/T", "CVT"], n),
        "ext_col": rng.choice(["Black", "White", "–"], n),
        "int_col": rng.choice(["Black", "Gray", "–"], n),
        "accident": rng.choice(["None reported", "At least 1 accident", np.nan], n),
        "clean_title": rng.choice(["Yes", np.nan], n),
        "price": [f"${p:,}" for p in rng.integers(10000, 80000, n)],
    })
    df = parse_raw(raw, reference_year=2026)
    X, y = df[FEATURES], df["price"]

    # 1) Ridge (scale=True): ikili eksiklik göstergesi ölçeklenmemeli (yalnızca 0 veya 1 olmalı)
    ridge_model = build_model(Ridge(), scale=True).fit(X, y)
    prep = ridge_model.regressor_[:-1]
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
    ind_cols = [c for c in Xt.columns if "missingindicator" in c]
    assert len(ind_cols) > 0
    for c in ind_cols:
        assert set( np.unique(Xt[c]) ).issubset({0.0, 1.0})

    # 2) XGBoost + SHAP (XGBoost 3.x base_score uyumluluğu)
    xgb_model = build_model(XGBRegressor(n_estimators=10, max_depth=3, random_state=42), scale=False).fit(X, y)
    xgb_prep = xgb_model.regressor_[:-1]
    xgb_reg = xgb_model.regressor_[-1]
    Xt_xgb = pd.DataFrame(xgb_prep.transform(X), columns=xgb_prep.get_feature_names_out())
    sv = compute_shap(xgb_reg, Xt_xgb.iloc[:10])
    assert sv.values.shape == (10, Xt_xgb.shape[1])

    # 3) predict_raw: eksik sütunlarla (milage/brand dahil) çökmeden tahmin üretmeli
    bundle = {"model": xgb_model, "reference_year": 2026, "interval_log_q90": 0.25, "features": FEATURES}
    partial_input = pd.DataFrame([{"brand": "Ford", "model": "F-150"}])
    pred_df = predict_raw(partial_input, bundle)
    assert list(pred_df.columns) == ["prediction", "low_90", "high_90"]
    assert (pred_df["low_90"] < pred_df["prediction"]).all()
    assert (pred_df["prediction"] < pred_df["high_90"]).all()
