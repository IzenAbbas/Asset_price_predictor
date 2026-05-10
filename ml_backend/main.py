"""FastAPI application — Asset Price Prediction API.

Endpoints:
  GET  /            → Status check
  GET  /health      → Detailed health with model load status
  GET  /options      → Dropdown options extracted from CSVs
  POST /predict/car  → Predict car price
  POST /predict/house→ Predict house price
  POST /extract/vehicle-fields → Extract vehicle fields from URL using NER
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schemas import (
    CarPredictionInput,
    CarPredictionOutput,
    DropdownOptions,
    HealthResponse,
    HousePredictionInput,
    HousePredictionOutput,
    HouseFieldsOutput,
    VehicleFieldsOutput,
)
from predictor import predictor

# Try to import NER functionality (modules are co-located in ml_backend/)
try:
    from ner_cars import extract_vehicle_fields
except ImportError:
    extract_vehicle_fields = None

try:
    from ner_houses import extract_house_fields
except ImportError:
    extract_house_fields = None

app = FastAPI(
    title="Asset Price Prediction API",
    description=(
        "REST API serving car and house price predictions for the Pakistani market. "
        "Models are trained on PakWheels and Zameen data."
    ),
    version="1.0.0",
)

# ── CORS — allow Flutter app (web, emulator, device) to reach this API ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock this down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health & Status ──────────────────────────────────────────────────

@app.get("/", tags=["status"])
def root():
    return {"status": "Asset Price Prediction API is running ✅"}


@app.get("/health", response_model=HealthResponse, tags=["status"])
def health_check():
    return HealthResponse(
        status="healthy",
        car_model_loaded=predictor.car_model is not None,
        house_model_loaded=predictor.house_model is not None,
    )


# ── Dropdown Options ─────────────────────────────────────────────────

@app.get("/options", response_model=DropdownOptions, tags=["options"])
def get_options():
    """Return dropdown option lists so the Flutter frontend can populate
    select boxes dynamically from the training CSV data."""
    try:
        opts = predictor.get_dropdown_options()
        return DropdownOptions(**opts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Prediction Endpoints ─────────────────────────────────────────────

@app.post("/predict/car", response_model=CarPredictionOutput, tags=["prediction"])
def predict_car(input_data: CarPredictionInput):
    try:
        result = predictor.predict_car_price(input_data.model_dump())
        return CarPredictionOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/house", response_model=HousePredictionOutput, tags=["prediction"])
def predict_house(input_data: HousePredictionInput):
    try:
        result = predictor.predict_house_price(input_data.model_dump())
        return HousePredictionOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Vehicle Field Extraction (NER from URL) ────────────────────────────

class VehicleURLInput(BaseModel):
    """Request body for vehicle field extraction from URL."""
    url: str


class HouseURLInput(BaseModel):
    """Request body for house field extraction from URL."""
    url: str


@app.post("/extract/vehicle-fields", response_model=VehicleFieldsOutput, tags=["extraction"])
def extract_vehicle_fields_endpoint(input_data: VehicleURLInput):
    """Extract vehicle fields (NER) from a PakWheels listing URL.
    
    Supported domains:
      - PakWheels: https://www.pakwheels.com/...
    """
    if not extract_vehicle_fields:
        raise HTTPException(
            status_code=503,
            detail="NER extraction not available. Missing ner_cars module or dependencies."
        )
    
    try:
        # Validate URL domain
        lower_url = input_data.url.lower()
        if "pakwheels" not in lower_url:
            raise HTTPException(
                status_code=400,
                detail="Unsupported URL domain. Only PakWheels is supported. "
                       "Example: https://www.pakwheels.com/..."
            )
        
        # Extract fields using NER
        result = extract_vehicle_fields(input_data.url)
        return VehicleFieldsOutput(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/extract/house-fields", response_model=HouseFieldsOutput, tags=["extraction"])
def extract_house_fields_endpoint(input_data: HouseURLInput):
    """Extract house fields (NER) from a Zameen listing URL.
    
    Supported domains:
      - Zameen: https://www.zameen.com/...
    """
    if not extract_house_fields:
        raise HTTPException(
            status_code=503,
            detail="NER extraction not available. Missing ner_houses module or dependencies."
        )

    try:
        lower_url = input_data.url.lower()
        if "zameen.com" not in lower_url:
            raise HTTPException(
                status_code=400,
                detail="Unsupported URL domain. Only Zameen is supported. "
                       "Example: https://www.zameen.com/..."
            )

        result = extract_house_fields(input_data.url)
        return HouseFieldsOutput(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
