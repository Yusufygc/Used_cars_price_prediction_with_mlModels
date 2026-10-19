"""Eğitim: split -> baseline -> CV ile model seçimi -> test seti TEK sefer -> kaydet.

Çalıştırma:
    python -m src.train          # varsayılan (data/used_cars.csv)
    python -m src.train --tr     # Türkiye veri seti (data/tr_used_cars.csv)
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (KFold, RandomizedSearchCV, RepeatedKFold,
                                     cross_val_predict, cross_val_score, train_test_split)
from sklearn.svm import SVR
from xgboost import XGBRegressor

from . import config as C
from .features import parse_raw
from .pipeline import build_model


def load_data(reference_year: int, data_path: Path = C.DATA_PATH) -> pd.DataFrame:
    raw = pd.read_csv(data_path)
    df = parse_raw(raw, reference_year)
    before = len(df)
    df = df.dropna(subset=[C.TARGET, "age", "mileage"]).reset_index(drop=True)
    print(f"Veri ({data_path.name}): {before} satır, hedef/yaş/mil eksik olan {before - len(df)} satır atıldı.")
    return df


def dollar_metrics(y_true, y_pred) -> dict:
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "r2_log": float(r2_score(np.log1p(y_true), np.log1p(y_pred))),
        "r2_dollar": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(ape.mean()),
        "median_ape": float(np.median(ape)),
    }


def candidates() -> dict:
    """model adı -> (regressor, scale?, arama uzayı)"""
    return {
        "ridge": (Ridge(), True, {"alpha": loguniform(1e-2, 1e2)}),
        "svr": (SVR(), True, {"C": loguniform(1, 100), "epsilon": uniform(0.01, 0.2),
                              "gamma": ["scale", "auto"]}),
        "random_forest": (RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1), False, {
            "n_estimators": randint(200, 500), "max_depth": [8, 12, 16, None],
            "min_samples_leaf": randint(1, 6), "max_features": uniform(0.3, 0.7)}),
        "xgboost": (XGBRegressor(random_state=C.RANDOM_STATE, n_jobs=-1, tree_method="hist"), False, {
            "n_estimators": randint(200, 700), "learning_rate": loguniform(0.02, 0.2),
            "max_depth": randint(3, 8), "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.5, 0.5), "reg_lambda": loguniform(0.1, 10),
            "min_child_weight": randint(1, 8)}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr", action="store_true", help="Türkiye veri setini (data/tr_used_cars.csv) kullan")
    ap.add_argument("--data", type=Path, default=None, help="Özel veri dosyası yolu")
    args = ap.parse_args()

    data_path = args.data if args.data else (C.TR_DATA_PATH if args.tr else C.DATA_PATH)
    C.MODEL_DIR.mkdir(exist_ok=True)
    C.REPORT_DIR.mkdir(exist_ok=True)
    reference_year = datetime.now().year
    df = load_data(reference_year, data_path)

    # 1) SPLIT İLK ADIM. Test seti bundan sonra sadece en sonda bir kez kullanılır.
    X, y = df[C.FEATURES], df[C.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

    inner_cv = KFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    cv = RepeatedKFold(n_splits=C.CV_FOLDS, n_repeats=C.CV_REPEATS, random_state=C.RANDOM_STATE + 1)

    # 2) Baseline'lar (CV, yalnızca train)
    rows = []
    base_defs = {
        "baseline_median": DummyRegressor(strategy="median"),
    }
    for name, reg in base_defs.items():
        s = cross_val_score(build_model(reg), X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error")
        rows.append({"model": name, "cv_mae": -s.mean(), "cv_mae_std": s.std()})

    # 3) Model seçimi: her aday için rastgele arama (inner_cv), sonra tekrarlı CV ile karşılaştırma
    tuned = {}
    for name, (reg, scale, space) in candidates().items():
        print(f"\n[{name}] hiperparametre araması...")
        search = RandomizedSearchCV(
            build_model(reg, scale), {f"regressor__reg__{k}": v for k, v in space.items()},
            n_iter=C.SEARCH_ITER, cv=inner_cv, scoring="neg_mean_absolute_error",
            random_state=C.RANDOM_STATE, n_jobs=1, refit=True)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        # Farklı fold bölmeli tekrarlı CV ile model karşılaştırma skoru
        s = cross_val_score(best, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error")
        tuned[name] = (best, search.best_params_)
        rows.append({"model": name, "cv_mae": -s.mean(), "cv_mae_std": s.std(),
                     "best_params": json.dumps({k.split("__")[-1]: v for k, v in search.best_params_.items()},
                                               default=float)})
        print(f"  CV MAE: ${-s.mean():,.0f} ± {s.std():,.0f}")

    cv_table = pd.DataFrame(rows).sort_values("cv_mae")
    cv_table.to_csv(C.REPORT_DIR / "cv_results.csv", index=False)
    print("\n=== CV sonuçları (train) ===\n", cv_table[["model", "cv_mae", "cv_mae_std"]].to_string(index=False))

    best_name = cv_table[cv_table.model.isin(tuned)].iloc[0]["model"]
    best_model, best_params = tuned[best_name]
    print(f"\nSeçilen model: {best_name}")

    # 4) Belirsizlik: train CV artıklarından log-uzayında %90 aralık genişliği
    oof = cross_val_predict(best_model, X_tr, y_tr, cv=inner_cv)
    log_resid = np.abs(np.log1p(y_tr) - np.log1p(oof))
    q90 = float(np.quantile(log_resid, 0.90))

    # 5) TEST SETİ: tek sefer, sadece seçilen model + baseline referansı
    pred = best_model.predict(X_te)
    test_metrics = dollar_metrics(y_te.to_numpy(), pred)
    base = build_model(DummyRegressor(strategy="median")).fit(X_tr, y_tr)
    base_metrics = dollar_metrics(y_te.to_numpy(), base.predict(X_te))
    lo = np.expm1(np.log1p(pred) - q90)
    hi = np.expm1(np.log1p(pred) + q90)
    test_metrics["interval90_coverage"] = float(((y_te.to_numpy() >= lo) & (y_te.to_numpy() <= hi)).mean())
    print("\n=== TEST (tek sefer) ===")
    print(json.dumps({"model": test_metrics, "baseline_median": base_metrics}, indent=2))

    # 6) Hata analizi
    err = X_te.copy()
    err["price"], err["pred"] = y_te.to_numpy(), pred
    err["ape"] = np.abs(err.pred - err.price) / err.price
    err["price_bucket"] = pd.qcut(err.price, 5, duplicates="drop").astype(str)
    err["age_bucket"] = pd.cut(err.age, [-1, 3, 7, 12, 20, 100]).astype(str)
    top_brands = err.brand.value_counts().head(10).index
    parts = []
    for col in ["price_bucket", "age_bucket"]:
        parts.append(err.groupby(col).agg(n=("ape", "size"), mape=("ape", "mean"),
                                          mae=("pred", lambda s: (s - err.loc[s.index, "price"]).abs().mean())
                                          ).assign(group=col))
    parts.append(err[err.brand.isin(top_brands)].groupby("brand").agg(
        n=("ape", "size"), mape=("ape", "mean"),
        mae=("pred", lambda s: (s - err.loc[s.index, "price"]).abs().mean())).assign(group="brand"))
    pd.concat(parts).to_csv(C.REPORT_DIR / "error_analysis.csv")
    err.to_csv(C.REPORT_DIR / "test_predictions.csv", index=False)

    # 7) Kaydet: sadece train ile fit edilmiş model kaydedilir; böylece test_predictions.csv ve XAI birebir tutarlıdır
    bg = X_tr.sample(min(100, len(X_tr)), random_state=C.RANDOM_STATE).reset_index(drop=True)
    bundle = {"model": best_model, "reference_year": reference_year, "interval_log_q90": q90,
              "model_name": best_name, "params": best_params, "features": C.FEATURES,
              "background": bg}
    joblib.dump(bundle, C.BUNDLE_PATH)
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(
        {"selected": best_name, "test": test_metrics, "baseline_median_test": base_metrics,
          "reference_year": reference_year}, indent=2), encoding="utf-8")
    print(f"\nKaydedildi: {C.BUNDLE_PATH}")


if __name__ == "__main__":
    main()
