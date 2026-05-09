# Asset Price Prediction API

REST API for predicting car and house prices in the Pakistani market. Models are trained on **PakWheels** (cars) and **Zameen** (houses) datasets.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Base URL](#base-url)
- [API Endpoints Overview](#api-endpoints-overview)
- [1. Status & Health](#1-status--health)
- [2. Dropdown Options](#2-dropdown-options)
- [3. Car Price Prediction](#3-car-price-prediction)
- [4. House Price Prediction](#4-house-price-prediction)
- [5. Vehicle Field Extraction (NER)](#5-vehicle-field-extraction-ner-from-url)
- [Error Handling](#error-handling)
- [CORS](#cors)
- [Interactive API Docs](#interactive-api-docs)
- [Frontend Integration Notes](#frontend-integration-notes)

---

## Quick Start

### 1. Install dependencies

```bash
cd ml_backend
pip install -r requirements.txt
```

### 2. Start the server

```bash
cd ml_backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Stop the server

```bash
# Find and kill the process on port 8000
lsof -ti:8000 | xargs kill
```

### 4. Verify it's running

```bash
curl http://127.0.0.1:8000/
# → {"status": "Asset Price Prediction API is running ✅"}
```

---

## Base URL

| Environment         | Base URL                     |
| ------------------- | ---------------------------- |
| Local development   | `http://127.0.0.1:8000`     |
| Android emulator    | `http://10.0.2.2:8000`      |
| iOS simulator       | `http://127.0.0.1:8000`     |
| Production (Render) | `https://your-app.onrender.com` |

All endpoints below are relative to the base URL.

---

## API Endpoints Overview

| Method | Endpoint                   | Description                         |
| ------ | -------------------------- | ----------------------------------- |
| GET    | `/`                        | Status check                        |
| GET    | `/health`                  | Health check with model load status |
| GET    | `/options`                 | Dropdown options for form fields    |
| POST   | `/predict/car`             | Predict car price                   |
| POST   | `/predict/house`           | Predict house price                 |
| POST   | `/extract/vehicle-fields`  | Extract car fields from listing URL |

---

## 1. Status & Health

### `GET /`

Simple status check to verify the API is running.

**Request:**
```bash
curl http://127.0.0.1:8000/
```

**Response** `200 OK`:
```json
{
  "status": "Asset Price Prediction API is running ✅"
}
```

---

### `GET /health`

Detailed health check showing whether ML models are loaded.

**Request:**
```bash
curl http://127.0.0.1:8000/health
```

**Response** `200 OK`:
```json
{
  "status": "healthy",
  "car_model_loaded": true,
  "house_model_loaded": true
}
```

| Field                | Type    | Description                          |
| -------------------- | ------- | ------------------------------------ |
| `status`             | string  | `"healthy"` if server is running     |
| `car_model_loaded`   | boolean | `true` if car model `.pkl` is loaded |
| `house_model_loaded` | boolean | `true` if house model `.pkl` is loaded |

---

## 2. Dropdown Options

### `GET /options`

Returns all valid dropdown values for both car and house prediction forms. **Call this once on app startup** and use the returned values to populate your dropdown/select fields.

**Request:**
```bash
curl http://127.0.0.1:8000/options
```

**Response** `200 OK`:
```json
{
  "car": {
    "fuel_types": ["cng", "diesel", "electric", "hybrid", "lpg", "petrol"],
    "transmissions": ["automatic", "manual"],
    "assemblies": ["imported", "local"],
    "brands": ["audi", "bmw", "changan", "chery", "daihatsu", "faw", "honda", "hyundai", "kia", "lexus", "mazda", "mercedes", "mg", "mitsubishi", "nissan", "peugeot", "prince", "subaru", "suzuki", "toyota", "united", "volkswagen"],
    "model_names": ["alto", "aqua", "baleno", "bolan", "brio", "city", "civic", "corolla", "corolla altis", "cultus", "fortuner", "hiace", "hilux", "land cruiser", "liana", "mehran", "mira", "passo", "picanto", "prado", "prius", "ravi", "sportage", "swift", "tucson", "vitz", "wagon r", "yaris"],
    "cities": ["faisalabad", "islamabad", "karachi", "lahore", "multan", "peshawar", "rawalpindi"]
  },
  "house": {
    "property_types": ["farm house", "flat", "house", "lower portion", "penthouse", "room", "upper portion"],
    "cities": ["faisalabad", "islamabad", "karachi", "lahore", "rawalpindi"],
    "provinces": ["islamabad capital", "punjab", "sindh"],
    "purposes": ["for rent", "for sale"]
  }
}
```

> **Note:** The actual values are extracted from the training CSV data and may vary slightly. Always use the values from this endpoint for your dropdowns — do NOT hardcode them.

### Response structure

| Field                    | Type         | Description                                    |
| ------------------------ | ------------ | ---------------------------------------------- |
| `car.fuel_types`         | `string[]`   | Valid fuel types (lowercase)                   |
| `car.transmissions`      | `string[]`   | Valid transmissions (lowercase)                |
| `car.assemblies`         | `string[]`   | Valid assembly types (lowercase)               |
| `car.brands`             | `string[]`   | Valid car brands (lowercase)                   |
| `car.model_names`        | `string[]`   | Valid model names (lowercase)                  |
| `car.cities`             | `string[]`   | Valid cities (lowercase)                       |
| `house.property_types`   | `string[]`   | Valid property types (lowercase)               |
| `house.cities`           | `string[]`   | Valid cities (lowercase)                       |
| `house.provinces`        | `string[]`   | Valid provinces (lowercase)                    |
| `house.purposes`         | `string[]`   | Valid purposes (lowercase)                     |

---

## 3. Car Price Prediction

### `POST /predict/car`

Predicts the price of a used car based on its features.

**Headers:**
```
Content-Type: application/json
```

**Request body:**

```json
{
  "model_year": 2020,
  "mileage": 50000,
  "engine_capacity": 1300,
  "fuel_type": "petrol",
  "transmission": "automatic",
  "assembly": "local",
  "brand": "honda",
  "model_name": "civic"
}
```

### Request fields

| Field              | Type    | Required | Constraints             | Description                                    |
| ------------------ | ------- | -------- | ----------------------- | ---------------------------------------------- |
| `model_year`       | integer | ✅       | `1980 ≤ value ≤ 2030`  | Manufacturing year of the vehicle              |
| `mileage`          | float   | ✅       | `≥ 0`                  | Mileage in **kilometres**                      |
| `engine_capacity`  | float   | ✅       | `≥ 0`                  | Engine capacity in **cc**                      |
| `fuel_type`        | string  | ✅       | From `/options`         | e.g. `"petrol"`, `"diesel"`, `"hybrid"`, `"cng"` |
| `transmission`     | string  | ✅       | From `/options`         | `"automatic"` or `"manual"`                    |
| `assembly`         | string  | ✅       | From `/options`         | `"local"` or `"imported"`                      |
| `brand`            | string  | ✅       | From `/options`         | Vehicle brand, **lowercase** (e.g. `"honda"`)  |
| `model_name`       | string  | ✅       | From `/options`         | Model name, **lowercase** (e.g. `"civic"`)     |

> **Important:** All string values must be **lowercase** and must match one of the values returned by `GET /options`.

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/predict/car \
  -H "Content-Type: application/json" \
  -d '{
    "model_year": 2020,
    "mileage": 50000,
    "engine_capacity": 1300,
    "fuel_type": "petrol",
    "transmission": "automatic",
    "assembly": "local",
    "brand": "honda",
    "model_name": "civic"
  }'
```

**Response** `200 OK`:
```json
{
  "predicted_price": 4520000.00,
  "formatted_price": "PKR 4,520,000"
}
```

| Field              | Type   | Description                                |
| ------------------ | ------ | ------------------------------------------ |
| `predicted_price`  | float  | Raw predicted price in PKR                 |
| `formatted_price`  | string | Human-readable formatted price string      |

---

## 4. House Price Prediction

### `POST /predict/house`

Predicts the price of a house/property based on its features.

**Headers:**
```
Content-Type: application/json
```

**Request body:**

```json
{
  "Total_Area": 10,
  "bedrooms": 3,
  "baths": 2,
  "latitude": 33.6844,
  "longitude": 73.0479,
  "listing_year": 2024,
  "listing_month": 6,
  "property_type": "house",
  "location": "dha phase 6",
  "city": "islamabad",
  "province_name": "islamabad capital",
  "purpose": "for sale"
}
```

### Request fields

| Field            | Type    | Required | Constraints              | Description                                    |
| ---------------- | ------- | -------- | ------------------------ | ---------------------------------------------- |
| `Total_Area`     | float   | ✅       | `≥ 0`                   | Total area in **marla / sq ft** (as used on Zameen) |
| `bedrooms`       | integer | ✅       | `0 ≤ value ≤ 50`        | Number of bedrooms                             |
| `baths`          | integer | ✅       | `0 ≤ value ≤ 50`        | Number of bathrooms                            |
| `latitude`       | float   | ✅       | `20.0 ≤ value ≤ 38.0`   | Latitude of the property                       |
| `longitude`      | float   | ✅       | `60.0 ≤ value ≤ 78.0`   | Longitude of the property                      |
| `listing_year`   | integer | ✅       | `2010 ≤ value ≤ 2030`   | Year the listing was created                   |
| `listing_month`  | integer | ✅       | `1 ≤ value ≤ 12`        | Month the listing was created                  |
| `property_type`  | string  | ✅       | From `/options`          | e.g. `"house"`, `"flat"`, `"room"`, `"upper portion"` |
| `location`       | string  | ✅       | From `/options`          | e.g. `"dha phase 6"`, `"g-10"`, `"johar town"` |
| `city`           | string  | ✅       | From `/options`          | e.g. `"karachi"`, `"lahore"`, `"islamabad"`    |
| `province_name`  | string  | ✅       | From `/options`          | e.g. `"sindh"`, `"punjab"`, `"islamabad capital"` |
| `purpose`        | string  | ✅       | From `/options`          | `"for sale"` or `"for rent"`                   |

> **Important:** All string values must be **lowercase**.

> **Note:** `Total_Area` has a **capital T** — this is intentional and matches the training data schema.

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/predict/house \
  -H "Content-Type: application/json" \
  -d '{
    "Total_Area": 10,
    "bedrooms": 3,
    "baths": 2,
    "latitude": 33.6844,
    "longitude": 73.0479,
    "listing_year": 2024,
    "listing_month": 6,
    "property_type": "house",
    "location": "dha phase 6",
    "city": "islamabad",
    "province_name": "islamabad capital",
    "purpose": "for sale"
  }'
```

**Response** `200 OK`:
```json
{
  "predicted_price": 32500000.00,
  "formatted_price": "PKR 32,500,000"
}
```

| Field              | Type   | Description                                |
| ------------------ | ------ | ------------------------------------------ |
| `predicted_price`  | float  | Raw predicted price in PKR                 |
| `formatted_price`  | string | Human-readable formatted price string      |

---

## 5. Vehicle Field Extraction (NER from URL)

### `POST /extract/vehicle-fields`

Scrapes a **PakWheels** or **OLX** car listing URL and automatically extracts vehicle fields using AI (NER). Use this to **auto-fill** the car prediction form from a listing URL.

**Headers:**
```
Content-Type: application/json
```

**Request body:**

```json
{
  "url": "https://www.pakwheels.com/used-cars/honda-civic-exi-prosmatec-2004-11465498"
}
```

### Request fields

| Field | Type   | Required | Description                              |
| ----- | ------ | -------- | ---------------------------------------- |
| `url` | string | ✅       | Full URL of a PakWheels or OLX listing   |

### Supported domains

| Domain     | Example URL                                                              |
| ---------- | ------------------------------------------------------------------------ |
| PakWheels  | `https://www.pakwheels.com/used-cars/honda-civic-exi-prosmatec-2004-11465498` |
| OLX        | `https://www.olx.com.pk/item/honda-civic-2020-iid-123456789`            |

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/extract/vehicle-fields \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.pakwheels.com/used-cars/honda-civic-exi-prosmatec-2004-11465498"
  }'
```

**Response** `200 OK`:
```json
{
  "model_year": 2004,
  "mileage_km": 160000,
  "engine_capacity_cc": 1500,
  "fuel_type": "petrol",
  "transmission": "automatic",
  "assembly": "local",
  "brand": "honda",
  "model_name": "civic"
}
```

### Response fields

| Field                | Type           | Description                                   |
| -------------------- | -------------- | --------------------------------------------- |
| `model_year`         | integer \| null | Extracted year (e.g. `2004`)                  |
| `mileage_km`         | float \| null   | Extracted mileage in km (e.g. `160000`)       |
| `engine_capacity_cc` | float \| null   | Extracted engine capacity in cc (e.g. `1500`) |
| `fuel_type`          | string \| null  | `"petrol"`, `"diesel"`, `"hybrid"`, etc.      |
| `transmission`       | string \| null  | `"automatic"` or `"manual"`                   |
| `assembly`           | string \| null  | `"local"` or `"imported"`                     |
| `brand`              | string \| null  | Brand name lowercase (e.g. `"honda"`)         |
| `model_name`         | string \| null  | Model name lowercase (e.g. `"civic"`)         |

> **Important:** Any field may be `null` if it could not be extracted. Your frontend should handle nulls gracefully — only auto-fill fields that have non-null values.

### Mapping NER output → Car prediction input

The NER response field names differ slightly from the car prediction input. Here's the mapping:

| NER Response Field     | Car Prediction Input Field | Notes                     |
| ---------------------- | -------------------------- | ------------------------- |
| `model_year`           | `model_year`               | Same name, direct map     |
| `mileage_km`           | `mileage`                  | ⚠️ Different name         |
| `engine_capacity_cc`   | `engine_capacity`          | ⚠️ Different name         |
| `fuel_type`            | `fuel_type`                | Same name, direct map     |
| `transmission`         | `transmission`             | Same name, direct map     |
| `assembly`             | `assembly`                 | Same name, direct map     |
| `brand`                | `brand`                    | Same name, direct map     |
| `model_name`           | `model_name`               | Same name, direct map     |

### Typical frontend flow

1. User pastes a PakWheels/OLX URL
2. Frontend calls `POST /extract/vehicle-fields` with the URL
3. Response returns extracted fields (some may be `null`)
4. Frontend auto-fills the prediction form with non-null values
5. User reviews/adjusts the auto-filled values
6. User submits the form → `POST /predict/car`

---

## Error Handling

All errors return a JSON response with a `detail` field:

```json
{
  "detail": "Error description here"
}
```

### HTTP Status Codes

| Code | Meaning               | When it happens                                                |
| ---- | --------------------- | -------------------------------------------------------------- |
| 200  | Success               | Request processed successfully                                 |
| 400  | Bad Request           | Invalid input (e.g. unsupported URL domain for NER)            |
| 422  | Validation Error      | Missing/invalid fields (Pydantic auto-validation)              |
| 500  | Internal Server Error | Server-side error (model failure, scraping error, etc.)        |
| 503  | Service Unavailable   | NER module not available (missing dependencies)                |

### 422 Validation Error format

When request validation fails, FastAPI returns a detailed error:

```json
{
  "detail": [
    {
      "loc": ["body", "model_year"],
      "msg": "Field required",
      "type": "missing"
    }
  ]
}
```

---

## CORS

The API has CORS enabled for **all origins** (`*`). Your frontend can call the API from any domain during development. No special headers needed beyond `Content-Type: application/json`.

---

## Interactive API Docs

FastAPI auto-generates interactive documentation:

| Tool       | URL                                  |
| ---------- | ------------------------------------ |
| Swagger UI | `http://127.0.0.1:8000/docs`        |
| ReDoc      | `http://127.0.0.1:8000/redoc`       |

You can **try out all endpoints** directly from the Swagger UI — no Postman needed.

---

## Frontend Integration Notes

### Recommended startup flow

```
1. App launches
2. GET /health → check if backend is up
3. GET /options → fetch dropdown values, cache them
4. Render forms with dropdown values from /options
5. On form submit → POST /predict/car or /predict/house
```

### Timeouts

| Endpoint                  | Recommended Timeout |
| ------------------------- | ------------------- |
| `GET /health`             | 5 seconds           |
| `GET /options`            | 10 seconds          |
| `POST /predict/car`       | 15 seconds          |
| `POST /predict/house`     | 15 seconds          |
| `POST /extract/vehicle-fields` | 30 seconds    |

> The NER endpoint (`/extract/vehicle-fields`) needs a longer timeout because it scrapes the listing page and calls Google Gemini AI for extraction.

### All strings must be lowercase

Every string field sent to `/predict/car` and `/predict/house` must be **lowercase**. The dropdown values from `/options` are already lowercase — just use them as-is.

### Handling null values from NER

When using `/extract/vehicle-fields` to auto-fill a form:

```
if (extracted.model_year != null) → set model year field
if (extracted.mileage_km != null) → set mileage field
if (extracted.fuel_type != null AND options.fuel_types.contains(extracted.fuel_type)) → set dropdown
// ... etc for each field
```

Always validate that the extracted value exists in your dropdown options before setting it.

---

## Data Sources

| Dataset                                        | Used For         | Source    |
| ---------------------------------------------- | ---------------- | --------- |
| `pakwheels_pakistan_automobile_dataset.csv`     | Car predictions  | PakWheels |
| `House_Details.csv`                            | House predictions | Zameen    |
