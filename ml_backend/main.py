
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
    EvaluationOutput,
)
from predictor import predictor

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/", tags=["status"])
def root():
    return {"status": "Asset Price Prediction API is running ✅"}

import json
import os

@app.get("/evaluation", response_model=EvaluationOutput, tags=["evaluation"])
def get_evaluation():
    try:
        base_dir = os.path.dirname(__file__)
        root_dir = os.path.dirname(base_dir)

        car_metrics_path = os.path.join(root_dir, "artifacts", "metrics.json")
        if not os.path.exists(car_metrics_path):
            car_metrics_path = os.path.join(base_dir, "artifacts", "metrics.json")

        with open(car_metrics_path, "r") as f:
            car_metrics = json.load(f)

        house_metrics_path = os.path.join(root_dir, "artifacts", "house_metrics.json")
        if not os.path.exists(house_metrics_path):
            house_metrics_path = os.path.join(base_dir, "artifacts", "house_metrics.json")

        with open(house_metrics_path, "r") as f:
            house_metrics = json.load(f)

        return EvaluationOutput(
            car_evaluation=car_metrics,
            house_evaluation=house_metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse, tags=["status"])
def health_check():
    return HealthResponse(
        status="healthy",
        car_model_loaded=predictor.car_model is not None,
        house_model_loaded=predictor.house_model is not None,
    )



@app.get("/options", response_model=DropdownOptions, tags=["options"])
def get_options():
    try:
        opts = predictor.get_dropdown_options()
        return DropdownOptions(**opts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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



class VehicleURLInput(BaseModel):
    url: str


class HouseURLInput(BaseModel):
    url: str


@app.post("/extract/vehicle-fields", response_model=VehicleFieldsOutput, tags=["extraction"])
def extract_vehicle_fields_endpoint(input_data: VehicleURLInput):
    if not extract_vehicle_fields:
        raise HTTPException(
            status_code=503,
            detail="NER extraction not available. Missing ner_cars module or dependencies."
        )

    try:
        lower_url = input_data.url.lower()
        if "pakwheels" not in lower_url:
            raise HTTPException(
                status_code=400,
                detail="Unsupported URL domain. Only PakWheels is supported. "
                       "Example: https://www.pakwheels.com/..."
            )

        result = extract_vehicle_fields(input_data.url)
        return VehicleFieldsOutput(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/extract/house-fields", response_model=HouseFieldsOutput, tags=["extraction"])
def extract_house_fields_endpoint(input_data: HouseURLInput):
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
