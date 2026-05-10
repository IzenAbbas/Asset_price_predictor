"""Model loading and prediction logic for car and house price models.

This module loads the trained sklearn pipelines from the artifacts/
folder once at import time and exposes a clean prediction interface
consumed by the FastAPI route handlers.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent  # ml_backend/
ARTIFACTS_DIR = BASE_DIR / "artifacts"

CAR_MODEL_PATH = ARTIFACTS_DIR / "car_price_best_model.pkl"
HOUSE_MODEL_PATH = ARTIFACTS_DIR / "house_price_best_model.pkl"
CAR_METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
HOUSE_METRICS_PATH = ARTIFACTS_DIR / "house_metrics.json"

CAR_DATA_PATH = BASE_DIR / "data" / "pakwheels_pakistan_automobile_dataset.csv"
HOUSE_DATA_PATH = BASE_DIR / "data" / "House_Details.csv"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_unique(df: pd.DataFrame, column: str, limit: int = 200) -> list[str]:
    """Return sorted unique lowercase values for a column (for dropdown options)."""
    if column not in df.columns:
        return []
    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .str.lower()
        .drop_duplicates()
        .head(limit)
        .tolist()
    )
    return sorted(values)


def _get_frequent_values(df: pd.DataFrame, column: str, min_count: int = 20, limit: int = 300) -> list[str]:
    """Return sorted unique values that appear at least min_count times.
    
    Useful for model names and other categorical fields where we want to include
    only reasonably common values to keep the dropdown manageable.
    """
    if column not in df.columns:
        return []
    
    # Count occurrences
    value_counts = df[column].value_counts()
    # Filter by minimum count
    frequent = value_counts[value_counts >= min_count].index.tolist()
    # Apply limit
    frequent = frequent[:limit]
    # Normalize to lowercase and sort
    return sorted([str(v).strip().lower() for v in frequent if pd.notna(v)])


def _get_brand_models(df: pd.DataFrame, min_count: int = 1) -> dict[str, list[str]]:
    """Return a dictionary mapping brands to their frequent model names."""
    if "brand" not in df.columns or "model_name" not in df.columns:
        return {}
        
    brand_models = {}
    for brand, group in df.groupby("brand"):
        brand_str = str(brand).strip().lower()
        if pd.isna(brand) or not brand_str:
            continue
            
        model_counts = group["model_name"].value_counts()
        frequent_models = model_counts[model_counts >= min_count].index.tolist()
        
        models = sorted([str(m).strip().lower() for m in frequent_models if pd.notna(m)])
        if models:
            brand_models[brand_str] = models
            
    if not brand_models:
        brand_models = {
            "toyota": ["corolla altis", "yaris", "fortuner", "hilux"],
            "honda": ["civic", "city", "br-v"],
            "suzuki": ["alto", "cultus", "wagon r", "swift"]
        }
    return brand_models


class MLPredictor:
    """Singleton-style ML predictor that holds both car and house models."""

    def __init__(self) -> None:
        # ── Load models ──
        if not CAR_MODEL_PATH.exists():
            raise FileNotFoundError(f"Car model not found: {CAR_MODEL_PATH}")
        if not HOUSE_MODEL_PATH.exists():
            raise FileNotFoundError(f"House model not found: {HOUSE_MODEL_PATH}")

        self.car_model = joblib.load(CAR_MODEL_PATH)
        self.house_model = joblib.load(HOUSE_MODEL_PATH)
        print("[OK] Car model loaded from", CAR_MODEL_PATH)
        print("[OK] House model loaded from", HOUSE_MODEL_PATH)

        # ── Load metrics (feature order) ──
        self.car_metrics = _load_json(CAR_METRICS_PATH)
        self.house_metrics = _load_json(HOUSE_METRICS_PATH)

        # ── Load datasets for dropdown options ──
        self.car_df = pd.read_csv(CAR_DATA_PATH)
        self.house_df = pd.read_csv(HOUSE_DATA_PATH)
        print(f"[OK] Car dataset: {len(self.car_df)} rows")
        print(f"[OK] House dataset: {len(self.house_df)} rows")

    # ── Dropdown options ──────────────────────────────────────────

    def get_dropdown_options(self) -> dict[str, dict[str, list[str]]]:
        return {
            "car": {
                "fuel_types": sorted(set((_safe_unique(self.car_df, "fuel_type") or ["petrol", "diesel", "hybrid"]) + ["electric"])),
                "transmissions": _safe_unique(self.car_df, "transmission") or ["automatic", "manual"],
                "assemblies": _safe_unique(self.car_df, "assembly") or ["local", "imported"],
                "brands": _safe_unique(self.car_df, "brand", limit=100) or ["toyota", "honda", "suzuki"],
                "model_names": _get_frequent_values(self.car_df, "model_name", min_count=20, limit=300) or ["corolla altis", "civic", "alto"],
                "cities": _safe_unique(self.car_df, "city"),
            },
            "house": {
                "property_types": _safe_unique(self.house_df, "property_type"),
                "cities": _safe_unique(self.house_df, "city"),
                "provinces": _safe_unique(self.house_df, "province_name"),
                "purposes": _safe_unique(self.house_df, "purpose"),
            },
            "car_brand_models": _get_brand_models(self.car_df)
        }

    # ── Car prediction ────────────────────────────────────────────

    def predict_car_price(self, data: dict[str, Any]) -> dict[str, Any]:
        feature_order = (
            self.car_metrics.get("features", {}).get("numeric", [])
            + self.car_metrics.get("features", {}).get("categorical", [])
        )
        current_year = datetime.now().year
        vehicle_age = max(0, current_year - int(data["model_year"]))

        feature_values = {
            "mileage": float(data["mileage"]),
            "engine_capacity": float(data["engine_capacity"]),
            "vehicle_age": float(vehicle_age),
            "fuel_type": str(data["fuel_type"]).strip(),
            "transmission": str(data["transmission"]).strip(),
            "assembly": str(data["assembly"]).strip(),
            "brand": str(data["brand"]).strip().lower(),
            "model_name": str(data["model_name"]).strip().lower(),
        }

        x = pd.DataFrame([{k: feature_values[k] for k in feature_order}])
        predicted = float(self.car_model.predict(x)[0])

        return {
            "predicted_price": round(predicted, 2),
            "formatted_price": f"PKR {predicted:,.0f}",
        }

    # ── House prediction ──────────────────────────────────────────

    def predict_house_price(self, data: dict[str, Any]) -> dict[str, Any]:
        feature_order = (
            self.house_metrics.get("features", {}).get("numeric", [])
            + self.house_metrics.get("features", {}).get("categorical", [])
        )

        feature_values = {
            "Total_Area": float(data["Total_Area"]),
            "bedrooms": float(data["bedrooms"]),
            "baths": float(data["baths"]),
            "latitude": float(data["latitude"]),
            "longitude": float(data["longitude"]),
            "listing_year": int(data["listing_year"]),
            "listing_month": int(data["listing_month"]),
            "property_type": str(data["property_type"]).strip().lower(),
            "location": str(data["location"]).strip().lower(),
            "city": str(data["city"]).strip().lower(),
            "province_name": str(data["province_name"]).strip().lower(),
            "purpose": str(data["purpose"]).strip().lower(),
        }

        x = pd.DataFrame([{k: feature_values[k] for k in feature_order}])
        predicted = float(self.house_model.predict(x)[0])

        return {
            "predicted_price": round(predicted, 2),
            "formatted_price": f"PKR {predicted:,.0f}",
        }


# ── Module-level singleton ────────────────────────────────────────────
predictor = MLPredictor()
