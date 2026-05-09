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
    VehicleFieldsOutput,
)
from predictor import predictor

# Try to import NER functionality
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ner_cars import extract_vehicle_fields
except ImportError:
    extract_vehicle_fields = None

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


@app.post("/extract/vehicle-fields", response_model=VehicleFieldsOutput, tags=["extraction"])
def extract_vehicle_fields_endpoint(input_data: VehicleURLInput):
    """Extract vehicle fields (NER) from a PakWheels or OLX listing URL.
    
    Supported domains:
      - PakWheels: https://www.pakwheels.com/...
      - OLX: https://www.olx.com.pk/...
    """
    if not extract_vehicle_fields:
        raise HTTPException(
            status_code=503,
            detail="NER extraction not available. Missing ner_cars module or dependencies."
        )
    
    try:
        # Validate URL domain
        lower_url = input_data.url.lower()
        if not ("pakwheels" in lower_url or "olx" in lower_url):
            raise HTTPException(
                status_code=400,
                detail="Unsupported URL domain. Only PakWheels and OLX are supported. "
                       "Example: https://www.pakwheels.com/... or https://www.olx.com.pk/..."
            )
        
        # Extract fields using NER
        result = extract_vehicle_fields(input_data.url)
        return VehicleFieldsOutput(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
