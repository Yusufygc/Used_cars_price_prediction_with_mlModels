"""SHAP: kaydedilmiş modeli (train.py çıktısı) açıklar. Açıklama log-fiyat uzayındadır
(SHAP değeri = log-fiyata katkı, exp(katkı) ≈ çarpan).

Çalıştırma: python -m src.explain
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from . import config as C


def compute_shap(reg, sample: pd.DataFrame, background: pd.DataFrame | None = None) -> shap.Explanation:
    """Ağaç ve doğrusal/kernel modeller için SHAP değerlerini hesaplar (XGBoost 3.x uyumlu)."""
    if hasattr(reg, "get_booster"):
        try:
            return shap.TreeExplainer(reg)(sample)
        except ValueError:
            contribs = reg.get_booster().predict(xgb.DMatrix(sample), pred_contribs=True)
            return shap.Explanation(
                values=contribs[:, :-1],
                base_values=contribs[:, -1],
                data=sample.to_numpy(),
                feature_names=list(sample.columns),
            )
    if hasattr(reg, "feature_importances_"):
        return shap.TreeExplainer(reg)(sample)
    bg = background if background is not None else sample.sample(min(100, len(sample)), random_state=0)
    return shap.Explainer(reg.predict, bg)(sample)


def main():
    C.REPORT_DIR.mkdir(exist_ok=True)
    bundle = joblib.load(C.BUNDLE_PATH)
    preds = pd.read_csv(C.REPORT_DIR / "test_predictions.csv")  # train.py'nin test kayıtları
    X = preds[C.FEATURES]

    ttr = bundle["model"]
    inner = ttr.regressor_            # Pipeline(prep, reg), log-uzayında çalışır
    prep, reg = inner[:-1], inner[-1]
    cols = prep.get_feature_names_out()
    Xt = pd.DataFrame(prep.transform(X), columns=cols, index=X.index)

    bg_raw = bundle.get("background")
    background = pd.DataFrame(prep.transform(bg_raw[C.FEATURES]), columns=cols) if bg_raw is not None \
        else Xt.sample(min(100, len(Xt)), random_state=0)

    sample = Xt.sample(min(400, len(Xt)), random_state=C.RANDOM_STATE)
    sv = compute_shap(reg, sample, background)

    shap.summary_plot(sv, sample, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(C.REPORT_DIR / "shap_summary.png", dpi=150); plt.close("all")

    mean_abs = pd.Series(np.abs(sv.values).mean(0), index=Xt.columns).sort_values(ascending=False)
    mean_abs.to_csv(C.REPORT_DIR / "shap_importance.csv", header=["mean_abs_shap"])
    for feat in mean_abs.index[:3]:
        shap.dependence_plot(feat, sv.values, sample, show=False)
        plt.tight_layout(); plt.savefig(C.REPORT_DIR / f"shap_dependence_{feat}.png", dpi=120); plt.close("all")

    # En büyük hatalı tahmin için waterfall (seçim veriden, elle index yok)
    worst_pos = int(preds.assign(e=(preds.pred - preds.price).abs()).e.to_numpy().argmax())
    sv_one = compute_shap(reg, Xt.iloc[[worst_pos]], background)
    shap.plots.waterfall(sv_one[0], show=False, max_display=15)
    plt.tight_layout(); plt.savefig(C.REPORT_DIR / "shap_waterfall_worst.png", dpi=150); plt.close("all")
    print("İlk 10 özellik (mean |SHAP|, log-fiyat):\n", mean_abs.head(10).round(4).to_string())
    print(f"Grafikler: {C.REPORT_DIR}")


if __name__ == "__main__":
    main()
