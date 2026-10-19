"""Tüm öğrenen adımlar (impute, scale, encode, hedef dönüşümü) tek Pipeline içinde.

Pipeline sadece train verisine fit edilir; CV içinde her fold kendi istatistiğini öğrenir.
Hedef log1p ile eğitilir, tahmin dolar olarak döner (TransformedTargetRegressor).
"""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from .config import HIGH_CARD, LOW_CARD, NUMERIC, RANDOM_STATE


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC),
            ("num_ind", MissingIndicator(features="missing-only", error_on_new=False), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                  sparse_output=False), LOW_CARD),
            # Yüksek kardinalite: iç CV'li target encoding (fit sırasında sızıntı yok).
            ("te", TargetEncoder(target_type="continuous", random_state=RANDOM_STATE), HIGH_CARD),
        ],
        verbose_feature_names_out=False,
    )


def build_model(regressor, scale: bool = True) -> TransformedTargetRegressor:
    """Ham özellik tablosu -> dolar tahmini."""
    inner = Pipeline([("prep", build_preprocessor(scale)), ("reg", regressor)])
    return TransformedTargetRegressor(regressor=inner, func=np.log1p, inverse_func=np.expm1)
