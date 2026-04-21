import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

try:
    from IPython.display import display  # type: ignore
except Exception:
    # Fallback for plain Python execution (non-notebook).
    def display(obj):  # type: ignore
        print(obj)


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "car_price_best_model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
HOUSE_MODEL_PATH = ARTIFACTS_DIR / "house_price_best_model.pkl"
HOUSE_METRICS_PATH = ARTIFACTS_DIR / "house_metrics.json"


def load_multicollinearity_decision(metrics_path: Path) -> str:
    if not metrics_path.exists():
        return "keep_all_numeric_features"

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics.get("multicollinearity_decision", "keep_all_numeric_features")


def load_room_feature_decision(metrics_path: Path) -> str:
    if not metrics_path.exists():
        return "keep_all_numeric_features"

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics.get("room_feature_decision", "keep_all_numeric_features")


def test_car_price_prediction(selected_model, multicollinearity_decision: str) -> pd.DataFrame:
    # Sample code to test Car Price Predictor
    x_sample = pd.DataFrame(
        {
            "mileage": [50000, 75000, 20000],
            "engine_capacity": [1300, 1500, 1000],
            "model_year": [2021, 2018, 2024],
            "fuel_type": ["Petrol", "Hybrid", "Petrol"],
            "transmission": ["Automatic", "Manual", "Automatic"],
            "assembly": ["Local", "Imported", "Local"],
            "brand": ["toyota", "honda", "suzuki"],
            "model_name": ["corolla altis", "civic", "alto"],
        }
    )

    current_year = datetime.now().year
    x_sample["vehicle_age"] = (current_year - x_sample["model_year"]).clip(lower=0)

    print("Sample Input Data:")
    display(x_sample)

    x_sample = x_sample.drop(columns=["model_year"])
    if multicollinearity_decision == "high_corr_detected_dropped_mileage_added_mileage_per_year":
        x_sample["mileage_per_year"] = x_sample["mileage"] / (x_sample["vehicle_age"] + 1)
        x_sample = x_sample.drop(columns=["mileage"])

    sample_predictions = selected_model.predict(x_sample)
    prediction_df = pd.DataFrame({"predicted_price": pd.Series(sample_predictions).round(2)})

    print("\nPredicted Prices for Sample Data:")
    display(prediction_df)
    return prediction_df


def test_house_price_prediction(selected_model, room_feature_decision: str) -> pd.DataFrame:
    # Sample records to verify house price prediction flow.
    x_sample = pd.DataFrame(
        {
            "Total_Area": [1089.004, 2450.0, 3560.0],
            "bedrooms": [2, 4, 5],
            "baths": [2, 4, 5],
            "latitude": [33.679890, 24.890000, 31.520400],
            "longitude": [73.012640, 67.070000, 74.358700],
            "listing_year": [2019, 2020, 2021],
            "listing_month": [2, 7, 11],
            "property_type": ["flat", "house", "house"],
            "location": ["g-10", "dha phase 6", "johar town"],
            "city": ["islamabad", "karachi", "lahore"],
            "province_name": ["islamabad capital", "sindh", "punjab"],
            "purpose": ["for sale", "for sale", "for sale"],
        }
    )

    print("\nHouse Sample Input Data:")
    display(x_sample)

    if room_feature_decision == "high_corr_detected_dropped_baths_added_bath_per_bedroom":
        x_sample["bath_per_bedroom"] = x_sample["baths"] / (x_sample["bedrooms"] + 1)
        x_sample = x_sample.drop(columns=["baths"])

    sample_predictions = selected_model.predict(x_sample)
    prediction_df = pd.DataFrame({"predicted_price": pd.Series(sample_predictions).round(2)})

    print("\nPredicted House Prices for Sample Data:")
    display(prediction_df)
    return prediction_df


def run_car_price_prediction_test() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    selected_model = joblib.load(MODEL_PATH)
    multicollinearity_decision = load_multicollinearity_decision(METRICS_PATH)
    print("Testing car price prediction:")
    test_car_price_prediction(selected_model, multicollinearity_decision)


def run_house_price_prediction_test() -> None:

    if not HOUSE_MODEL_PATH.exists():
        print(f"\nHouse model file not found: {HOUSE_MODEL_PATH}")
        print("Run house_price_prediction.ipynb first to generate house artifacts.")
        return

    house_model = joblib.load(HOUSE_MODEL_PATH)
    room_feature_decision = load_room_feature_decision(HOUSE_METRICS_PATH)
    print("\nTesting house price prediction:")
    test_house_price_prediction(house_model, room_feature_decision)


def main() -> None:
    run_car_price_prediction_test()
    run_house_price_prediction_test()


if __name__ == "__main__":
    main()
