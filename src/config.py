"""Proje sabitleri. Tek yerde tutulur."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "used_cars.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
BUNDLE_PATH = MODEL_DIR / "best_pipeline.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 2
SEARCH_ITER = 12

TARGET = "price"
NUMERIC = ["mileage", "age", "horsepower", "engine_volume", "cylinders"]
LOW_CARD = ["fuel_category", "transmission_type", "accident", "clean_title", "ext_col", "int_col"]
HIGH_CARD = ["brand", "model"]
FEATURES = NUMERIC + LOW_CARD + HIGH_CARD

EV_BRANDS = {"Tesla", "Rivian", "Lucid", "Polestar", "Fisker", "Karma"}
COLORS = ["black", "white", "gray", "silver", "blue", "red", "brown",
          "green", "beige", "gold", "yellow", "orange"]
