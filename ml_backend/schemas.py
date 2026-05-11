"""Pydantic request/response schemas for the Asset Price Prediction API.

The field names and types here mirror exactly what the trained sklearn
pipelines expect (see metrics.json / house_metrics.json for feature lists).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Car Price Prediction ──────────────────────────────────────────────

class CarPredictionInput(BaseModel):
    """Input features for car price prediction.

    Numeric: mileage, engine_capacity, vehicle_age (derived from model_year).
    Categorical: fuel_type, transmission, assembly, brand, model_name.
    """
    model_year: int = Field(..., ge=1980, le=2030, description="Manufacturing year of the vehicle")
    mileage: float = Field(..., ge=0, description="Mileage in kilometres")
    engine_capacity: float = Field(..., ge=0, description="Engine capacity in cc")
    fuel_type: str = Field(..., description="e.g. petrol, diesel, hybrid, cng, lpg")
    transmission: str = Field(..., description="e.g. automatic, manual")
    assembly: str = Field(..., description="e.g. local, imported")
    brand: str = Field(..., description="Vehicle brand (lowercase)")
    model_name: str = Field(..., description="Vehicle model name (lowercase)")


class CarPredictionOutput(BaseModel):
    predicted_price: float
    formatted_price: str


# ── Vehicle Field Extraction (NER from URL) ───────────────────────────

class VehicleFieldsOutput(BaseModel):
    """Extracted vehicle fields from a car listing URL using NER."""
    model_year: int | None = Field(None, description="Extracted model year")
    mileage_km: float | None = Field(None, description="Extracted mileage in km")
    engine_capacity_cc: float | None = Field(None, description="Extracted engine capacity in cc")
    fuel_type: str | None = Field(None, description="Extracted fuel type (lowercase)")
    transmission: str | None = Field(None, description="Extracted transmission type (lowercase)")
    assembly: str | None = Field(None, description="Extracted assembly type (lowercase)")
    brand: str | None = Field(None, description="Extracted vehicle brand (lowercase)")
    model_name: str | None = Field(None, description="Extracted model name (lowercase)")


# ── House Field Extraction (NER from URL) ───────────────────────────

class HouseFieldsOutput(BaseModel):
    """Extracted house fields from a Zameen listing URL using NER."""
    Total_Area: float | None = Field(None, description="Extracted total area")
    area_value: float | None = Field(None, description="Original area value in extracted unit")
    area_unit: str | None = Field(None, description="Original area unit")
    bedrooms: int | None = Field(None, description="Extracted bedroom count")
    baths: int | None = Field(None, description="Extracted bathroom count")
    latitude: float | None = Field(None, description="Extracted latitude")
    longitude: float | None = Field(None, description="Extracted longitude")
    listing_year: int | None = Field(None, description="Extracted listing year")
    listing_month: int | None = Field(None, description="Extracted listing month")
    property_type: str | None = Field(None, description="Extracted property type (lowercase)")
    location: str | None = Field(None, description="Extracted location (lowercase)")
    city: str | None = Field(None, description="Extracted city (lowercase)")
    province_name: str | None = Field(None, description="Extracted province (lowercase)")
    purpose: str | None = Field(None, description="Extracted purpose (lowercase)")


# ── House Price Prediction ────────────────────────────────────────────

class HousePredictionInput(BaseModel):
    """Input features for house price prediction.

    Numeric: Total_Area, bedrooms, baths, latitude, longitude,
             listing_year, listing_month.
    Categorical: property_type, location, city, province_name, purpose.
    """
    Total_Area: float = Field(..., ge=0, description="Total area (sq ft / marla)")
    bedrooms: int = Field(..., ge=0, le=50, description="Number of bedrooms")
    baths: int = Field(..., ge=0, le=50, description="Number of bathrooms")
    latitude: float = Field(..., ge=20.0, le=38.0, description="Latitude")
    longitude: float = Field(..., ge=60.0, le=78.0, description="Longitude")
    listing_year: int = Field(..., ge=2010, le=2030, description="Year the listing was created")
    listing_month: int = Field(..., ge=1, le=12, description="Month the listing was created")
    property_type: str = Field(..., description="e.g. house, flat, room, upper portion, lower portion, farm house, penthouse")
    location: str = Field(..., description="e.g. dha phase 6, g-10, johar town")
    city: str = Field(..., description="e.g. karachi, lahore, islamabad")
    province_name: str = Field(..., description="e.g. sindh, punjab, islamabad capital")
    purpose: str = Field(..., description="e.g. for sale, for rent")


class HousePredictionOutput(BaseModel):
    predicted_price: float
    formatted_price: str


# ── Dropdown Options (sent to frontend so it can populate selectors) ──

class DropdownOptions(BaseModel):
    car: dict[str, list[str]]
    house: dict[str, list[str]]
    car_brand_models: dict[str, list[str]] = Field(default_factory=dict)


# ── Health / Status ──────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    car_model_loaded: bool
    house_model_loaded: bool

# ── Evaluation ──────────────────────────────────────────────────

class EvaluationMetrics(BaseModel):
    selected_model: str
    test_mae: float
    test_rmse: float
    test_r2: float
    features: dict
    artifacts: dict
    graphs: dict = Field(default_factory=dict)

class EvaluationOutput(BaseModel):
    car_evaluation: EvaluationMetrics
    house_evaluation: EvaluationMetrics

