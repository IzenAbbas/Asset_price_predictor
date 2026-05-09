"""FastAPI application — Asset Price Prediction API.

Endpoints:
  GET  /            → Status check
  GET  /health      → Detailed health with model load status
  GET  /options      → Dropdown options extracted from CSVs
  POST /predict/car  → Predict car price
  POST /predict/house→ Predict house price
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    CarPredictionInput,
    CarPredictionOutput,
    DropdownOptions,
    HealthResponse,
    HousePredictionInput,
    HousePredictionOutput,
)
from predictor import predictor

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
