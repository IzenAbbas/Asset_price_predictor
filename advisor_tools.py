from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pandas as pd
from livekit.agents import function_tool


BASE_DIR = Path(__file__).resolve().parent
CAR_CSV_PATH = BASE_DIR / "pakwheels_pakistan_automobile_dataset.csv"
HOUSE_CSV_PATH = BASE_DIR / "House_Details.csv"


def _clamp_limit(limit: int | None, default: int = 5, maximum: int = 10) -> int:
    try:
        value = default if limit is None else int(limit)
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, maximum))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return cast(pd.DataFrame, pd.read_csv(path))  # type: ignore[call-overload]


@lru_cache(maxsize=1)
def load_car_data() -> pd.DataFrame:
    return _load_csv(CAR_CSV_PATH)


@lru_cache(maxsize=1)
def load_house_data() -> pd.DataFrame:
    return _load_csv(HOUSE_CSV_PATH)


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _records(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    return [{k: _clean_value(v) for k, v in row.items()} for row in df.head(limit).to_dict(orient="records")]


def _normalize_text(value: Any) -> str:
    return str(value).strip().casefold()


def _text_filter(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    if value is None or value == "":
        return df
    if column not in df.columns:
        return df.iloc[0:0]
    return df[df[column].astype(str).str.strip().str.casefold() == _normalize_text(value)]


def _numeric_range_filter(
    df: pd.DataFrame,
    column: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> pd.DataFrame:
    if column not in df.columns:
        return df.iloc[0:0]
    series = pd.to_numeric(df[column], errors="coerce")
    mask = series.notna()
    if minimum is not None:
        mask &= series >= minimum
    if maximum is not None:
        mask &= series <= maximum
    return df[mask]


def _price_summary(df: pd.DataFrame, price_column: str = "price") -> dict[str, Any]:
    if price_column not in df.columns:
        return {"count": 0}
    prices = pd.to_numeric(df[price_column], errors="coerce").dropna()
    if prices.empty:
        return {"count": 0}
    return {
        "count": int(prices.shape[0]),
        "min": float(prices.min()),
        "max": float(prices.max()),
        "mean": round(float(prices.mean()), 2),
        "median": round(float(prices.median()), 2),
    }


def _top_values(df: pd.DataFrame, column: str, limit: int = 5) -> list[dict[str, Any]]:
    if column not in df.columns:
        return []
    counts = df[column].astype(str).str.strip().replace("nan", pd.NA).dropna().value_counts().head(limit)
    return [{"value": _clean_value(idx), "count": int(count)} for idx, count in counts.items()]


def _dataset_overview(df: pd.DataFrame, label: str, limit: int) -> dict[str, Any]:
    sample_size = _clamp_limit(limit, default=5, maximum=10)
    return {
        "dataset": label,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "price_summary": _price_summary(df),
        "top_city_values": _top_values(df, "city"),
        "sample_rows": _records(df, sample_size),
    }


@function_tool(
    description="Inspect the local car CSV and return a compact overview with price statistics and sample rows."
)
async def get_car_dataset_overview(limit: int = 5) -> dict[str, Any]:
    return _dataset_overview(load_car_data(), "pakwheels_pakistan_automobile_dataset.csv", _clamp_limit(limit))


@function_tool(
    description=(
        "Search the local car CSV using exact text filters and numeric ranges. "
        "Useful for finding comparable used-car listings and summarizing their prices."
    )
)
async def search_car_listings(
    city: str | None = None,
    model: str | None = None,
    fuel_type: str | None = None,
    transmission: str | None = None,
    assembly: str | None = None,
    registered: str | None = None,
    color: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    mileage_min: float | None = None,
    mileage_max: float | None = None,
    engine_capacity_min: float | None = None,
    engine_capacity_max: float | None = None,
    vehicle_age_min: float | None = None,
    vehicle_age_max: float | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    df = load_car_data().copy()
    df = _text_filter(df, "city", city)
    df = _text_filter(df, "model", model)
    df = _text_filter(df, "fuel_type", fuel_type)
    df = _text_filter(df, "transmission", transmission)
    df = _text_filter(df, "assembly", assembly)
    df = _text_filter(df, "registered", registered)
    df = _text_filter(df, "color", color)
    df = _numeric_range_filter(df, "price", price_min, price_max)
    df = _numeric_range_filter(df, "mileage", mileage_min, mileage_max)
    df = _numeric_range_filter(df, "engine_capacity", engine_capacity_min, engine_capacity_max)
    df = _numeric_range_filter(df, "vehicle_age", vehicle_age_min, vehicle_age_max)

    sample_size = _clamp_limit(limit)
    return {
        "dataset": "pakwheels_pakistan_automobile_dataset.csv",
        "match_count": int(len(df)),
        "applied_filters": {
            "city": city,
            "model": model,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "assembly": assembly,
            "registered": registered,
            "color": color,
            "price_min": price_min,
            "price_max": price_max,
            "mileage_min": mileage_min,
            "mileage_max": mileage_max,
            "engine_capacity_min": engine_capacity_min,
            "engine_capacity_max": engine_capacity_max,
            "vehicle_age_min": vehicle_age_min,
            "vehicle_age_max": vehicle_age_max,
        },
        "price_summary": _price_summary(df),
        "top_cities": _top_values(df, "city"),
        "top_fuel_types": _top_values(df, "fuel_type"),
        "sample_rows": _records(df, sample_size),
    }


@function_tool(
    description="Inspect the local house CSV and return a compact overview with price statistics and sample rows."
)
async def get_house_dataset_overview(limit: int = 5) -> dict[str, Any]:
    return _dataset_overview(load_house_data(), "House_Details.csv", _clamp_limit(limit))


@function_tool(
    description=(
        "Search the local house CSV using exact text filters and numeric ranges. "
        "Useful for finding comparable property listings and summarizing their prices."
    )
)
async def search_house_listings(
    city: str | None = None,
    location: str | None = None,
    province_name: str | None = None,
    property_type: str | None = None,
    purpose: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    total_area_min: float | None = None,
    total_area_max: float | None = None,
    bedrooms_min: float | None = None,
    bedrooms_max: float | None = None,
    baths_min: float | None = None,
    baths_max: float | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    df = load_house_data().copy()
    df = _text_filter(df, "city", city)
    df = _text_filter(df, "location", location)
    df = _text_filter(df, "province_name", province_name)
    df = _text_filter(df, "property_type", property_type)
    df = _text_filter(df, "purpose", purpose)
    df = _numeric_range_filter(df, "price", price_min, price_max)
    df = _numeric_range_filter(df, "Total_Area", total_area_min, total_area_max)
    df = _numeric_range_filter(df, "bedrooms", bedrooms_min, bedrooms_max)
    df = _numeric_range_filter(df, "baths", baths_min, baths_max)

    sample_size = _clamp_limit(limit)
    return {
        "dataset": "House_Details.csv",
        "match_count": int(len(df)),
        "applied_filters": {
            "city": city,
            "location": location,
            "province_name": province_name,
            "property_type": property_type,
            "purpose": purpose,
            "price_min": price_min,
            "price_max": price_max,
            "total_area_min": total_area_min,
            "total_area_max": total_area_max,
            "bedrooms_min": bedrooms_min,
            "bedrooms_max": bedrooms_max,
            "baths_min": baths_min,
            "baths_max": baths_max,
        },
        "price_summary": _price_summary(df),
        "top_cities": _top_values(df, "city"),
        "top_property_types": _top_values(df, "property_type"),
        "sample_rows": _records(df, sample_size),
    }


CSV_TOOLS = [
    get_car_dataset_overview,
    search_car_listings,
    get_house_dataset_overview,
    search_house_listings,
]


