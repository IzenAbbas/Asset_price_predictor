from __future__ import annotations

import asyncio
import io
import inspect
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
# Load .env from project root so API keys are available to os.getenv calls.
load_dotenv(BASE_DIR / ".env")
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CAR_MODEL_PATH = ARTIFACTS_DIR / "car_price_best_model.pkl"
HOUSE_MODEL_PATH = ARTIFACTS_DIR / "house_price_best_model.pkl"
CAR_METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
HOUSE_METRICS_PATH = ARTIFACTS_DIR / "house_metrics.json"
CAR_DATA_PATH = BASE_DIR / "pakwheels_pakistan_automobile_dataset.csv"
HOUSE_DATA_PATH = BASE_DIR / "House_Details.csv"
ADVISOR_PATH = BASE_DIR / "advisor.py"


DETAILED_SYSTEM_PROMPT = """
You are POPO, a focused asset advisor for Pakistani used cars and residential properties.

Mission:
- Ground every factual statement in local CSV-backed data.
- Use only these datasets as your evidence base:
  - pakwheels_pakistan_automobile_dataset.csv for car advising.
  - House_Details.csv for house/property advising.

Behavior rules:
- Never fabricate numbers, trends, confidence scores, or unseen listing details.
- For valuation or comparison requests, first gather missing constraints.
- For cars, prioritize city, model, fuel type, transmission, assembly, registered city,
  color, mileage, engine capacity, and vehicle age/year.
- For houses, prioritize city, location, province, property type, purpose, bedrooms,
  baths, total area, and budget.
- If no records match, state that clearly and suggest how to relax one filter.
- If records are sparse, communicate uncertainty and avoid overclaiming precision.
- Keep guidance practical, concise, and PKR-focused.

Scope rules:
- If user asks outside asset advisory, politely steer back to car/house decisions.
- Explain recommendations using comparable listings and compact price summaries.
""".strip()


@st.cache_resource
def load_models() -> tuple[Any, Any]:
    if not CAR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Car model not found: {CAR_MODEL_PATH}")
    if not HOUSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"House model not found: {HOUSE_MODEL_PATH}")

    return joblib.load(CAR_MODEL_PATH), joblib.load(HOUSE_MODEL_PATH)


@st.cache_data
def load_metrics() -> tuple[dict[str, Any], dict[str, Any]]:
    with CAR_METRICS_PATH.open("r", encoding="utf-8") as f:
        car_metrics = json.load(f)
    with HOUSE_METRICS_PATH.open("r", encoding="utf-8") as f:
        house_metrics = json.load(f)
    return car_metrics, house_metrics


@st.cache_data
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    car_df = pd.read_csv(CAR_DATA_PATH)
    house_df = pd.read_csv(HOUSE_DATA_PATH)
    return car_df, house_df


@st.cache_data
def load_advisor_prompt_from_file() -> str:
    if not ADVISOR_PATH.exists():
        return DETAILED_SYSTEM_PROMPT

    text = ADVISOR_PATH.read_text(encoding="utf-8")
    match = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', text, flags=re.DOTALL)
    if not match:
        return DETAILED_SYSTEM_PROMPT

    prompt = match.group(1).strip()
    return prompt if prompt else DETAILED_SYSTEM_PROMPT


@st.cache_resource
def load_advisor_tools() -> tuple[Any, Any, str | None]:
    try:
        from advisor_tools import search_car_listings, search_house_listings

        return search_car_listings, search_house_listings, None
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, None, str(exc)


def invoke_tool(tool_callable: Any, **kwargs: Any) -> dict[str, Any]:
    result = tool_callable(**kwargs)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def fmt_pkr(value: float) -> str:
    return f"PKR {value:,.0f}"


def _safe_unique(df: pd.DataFrame, column: str, limit: int = 200) -> list[str]:
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


def build_car_features(car_metrics: dict[str, Any], user_input: dict[str, Any]) -> pd.DataFrame:
    feature_order = car_metrics.get("features", {}).get("numeric", []) + car_metrics.get("features", {}).get("categorical", [])
    current_year = datetime.now().year
    vehicle_age = max(0, current_year - int(user_input["model_year"]))

    feature_values = {
        "mileage": float(user_input["mileage"]),
        "engine_capacity": float(user_input["engine_capacity"]),
        "vehicle_age": float(vehicle_age),
        "fuel_type": str(user_input["fuel_type"]).strip(),
        "transmission": str(user_input["transmission"]).strip(),
        "assembly": str(user_input["assembly"]).strip(),
        "brand": str(user_input["brand"]).strip().lower(),
        "model_name": str(user_input["model_name"]).strip().lower(),
    }

    return pd.DataFrame([{k: feature_values[k] for k in feature_order}])


def build_house_features(house_metrics: dict[str, Any], user_input: dict[str, Any]) -> pd.DataFrame:
    feature_order = house_metrics.get("features", {}).get("numeric", []) + house_metrics.get("features", {}).get("categorical", [])

    feature_values = {
        "Total_Area": float(user_input["Total_Area"]),
        "bedrooms": float(user_input["bedrooms"]),
        "baths": float(user_input["baths"]),
        "latitude": float(user_input["latitude"]),
        "longitude": float(user_input["longitude"]),
        "listing_year": int(user_input["listing_year"]),
        "listing_month": int(user_input["listing_month"]),
        "property_type": str(user_input["property_type"]).strip().lower(),
        "location": str(user_input["location"]).strip().lower(),
        "city": str(user_input["city"]).strip().lower(),
        "province_name": str(user_input["province_name"]).strip().lower(),
        "purpose": str(user_input["purpose"]).strip().lower(),
    }

    return pd.DataFrame([{k: feature_values[k] for k in feature_order}])


def extract_budget(query: str) -> float | None:
    match = re.search(r"(?:under|below|max|budget|upto|up to)\s*(\d[\d,]*)", query.lower())
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def assistant_answer(
    query: str,
    car_cities: list[str],
    house_cities: list[str],
    search_car_listings: Any,
    search_house_listings: Any,
) -> str:
    q = query.strip().lower()
    if not q:
        return "Please enter a car or house advisory question."

    budget = extract_budget(q)

    if any(token in q for token in ["car", "vehicle", "pakwheels", "mileage", "engine"]):
        matched_city = next((c for c in car_cities if c in q), None)
        args = {"city": matched_city, "price_max": budget, "limit": 5}
        result = invoke_tool(search_car_listings, **args)
        count = result.get("match_count", 0)
        summary = result.get("price_summary", {})
        if count == 0:
            return "I could not find matching car listings in the local CSV. Try relaxing city or budget constraints."
        return (
            f"I found {count} comparable car listings from local data. "
            f"Observed range: {fmt_pkr(summary.get('min', 0))} to {fmt_pkr(summary.get('max', 0))}, "
            f"median {fmt_pkr(summary.get('median', 0))}."
        )

    if any(token in q for token in ["house", "property", "home", "plot", "bedroom", "baths"]):
        matched_city = next((c for c in house_cities if c in q), None)
        args = {"city": matched_city, "price_max": budget, "purpose": "For Sale", "limit": 5}
        result = invoke_tool(search_house_listings, **args)
        count = result.get("match_count", 0)
        summary = result.get("price_summary", {})
        if count == 0:
            return "I could not find matching house listings in the local CSV. Try relaxing city, purpose, or budget filters."
        return (
            f"I found {count} comparable house listings from local data. "
            f"Observed range: {fmt_pkr(summary.get('min', 0))} to {fmt_pkr(summary.get('max', 0))}, "
            f"median {fmt_pkr(summary.get('median', 0))}."
        )

    return (
        "I focus on car and residential property advice only. "
        "Please ask about a car or house with details like city, budget, mileage/area, and purpose."
    )


def transcribe_audio_with_openai(audio_bytes: bytes) -> tuple[str | None, str | None]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "Set OPENAI_API_KEY to enable voice-to-text transcription."

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None, "Install the OpenAI SDK to enable voice-to-text: pip install openai"

    try:
        client = OpenAI(api_key=api_key)
        file_obj = io.BytesIO(audio_bytes)
        file_obj.name = "voice_input.wav"
        transcript = client.audio.transcriptions.create(model="whisper-1", file=file_obj)
        text = getattr(transcript, "text", "")
        cleaned = str(text).strip()
        if not cleaned:
            return None, "No speech detected. Please try speaking again."
        return cleaned, None
    except Exception as exc:  # pragma: no cover - network/service dependent
        return None, f"Transcription failed: {exc}"


def speak_text_in_browser(text: str, lang: str = "en-US") -> None:
    payload = json.dumps(text)
    voice_lang = json.dumps(lang)
    components.html(
        f"""
        <script>
        const text = {payload};
        const lang = {voice_lang};
        if (text && window.speechSynthesis) {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = lang;
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }}
        </script>
        """,
        height=0,
    )


def main() -> None:
    st.set_page_config(page_title="Asset Advisor", page_icon="🏠", layout="wide")

    st.markdown(
        """
        <style>
        .fab-link {
            position: fixed;
            right: 24px;
            bottom: 24px;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: #2f80ed;
            color: white !important;
            font-size: 28px;
            line-height: 56px;
            text-align: center;
            text-decoration: none;
            box-shadow: 0 8px 20px rgba(0,0,0,0.28);
            z-index: 9999;
        }
        .fab-link:hover {
            background: #1f6ad0;
        }
        </style>
        <a class="fab-link" href="#ai-assistant" title="AI Assistant">💬</a>
        """,
        unsafe_allow_html=True,
    )

    st.title("Asset Advisor - Car and House Price Prediction")
    st.caption("Single-page dashboard for car and property valuation with a local-data AI assistant.")

    car_model, house_model = load_models()
    car_metrics, house_metrics = load_metrics()
    car_df, house_df = load_datasets()

    car_cities = _safe_unique(car_df, "city")
    house_cities = _safe_unique(house_df, "city")
    car_brands = _safe_unique(car_df, "brand")
    car_models = _safe_unique(car_df, "model_name")
    fuel_types = _safe_unique(car_df, "fuel_type")
    transmissions = _safe_unique(car_df, "transmission")
    assemblies = _safe_unique(car_df, "assembly")

    property_types = _safe_unique(house_df, "property_type")
    provinces = _safe_unique(house_df, "province_name")
    purposes = _safe_unique(house_df, "purpose")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Car Price Prediction")
        with st.form("car_form"):
            model_year = st.number_input("Model Year", min_value=1980, max_value=datetime.now().year, value=2020, step=1)
            mileage = st.number_input("Mileage (km)", min_value=0, value=50000, step=1000)
            engine_capacity = st.number_input("Engine Capacity (cc)", min_value=600, value=1300, step=100)
            fuel_type = st.selectbox("Fuel Type", options=fuel_types or ["petrol"])
            transmission = st.selectbox("Transmission", options=transmissions or ["automatic"])
            assembly = st.selectbox("Assembly", options=assemblies or ["local"])
            brand = st.selectbox("Brand", options=car_brands or ["toyota"])
            model_name = st.selectbox("Model Name", options=car_models or ["corolla altis"])

            predict_car = st.form_submit_button("Predict Car Price")

        if predict_car:
            x_car = build_car_features(
                car_metrics,
                {
                    "model_year": model_year,
                    "mileage": mileage,
                    "engine_capacity": engine_capacity,
                    "fuel_type": fuel_type,
                    "transmission": transmission,
                    "assembly": assembly,
                    "brand": brand,
                    "model_name": model_name,
                },
            )
            predicted = float(car_model.predict(x_car)[0])
            st.success(f"Predicted car price: {fmt_pkr(predicted)}")

    with col2:
        st.subheader("House Price Prediction")
        with st.form("house_form"):
            total_area = st.number_input("Total Area", min_value=50.0, value=1089.0, step=10.0)
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)
            baths = st.number_input("Baths", min_value=0, max_value=20, value=3, step=1)
            latitude = st.number_input("Latitude", min_value=20.0, max_value=38.0, value=33.6844, step=0.0001, format="%.6f")
            longitude = st.number_input("Longitude", min_value=60.0, max_value=78.0, value=73.0479, step=0.0001, format="%.6f")
            listing_year = st.number_input("Listing Year", min_value=2010, max_value=datetime.now().year, value=2022, step=1)
            listing_month = st.number_input("Listing Month", min_value=1, max_value=12, value=6, step=1)
            property_type = st.selectbox("Property Type", options=property_types or ["house"])
            location = st.text_input("Location", value="dha phase 6")
            city = st.selectbox("City", options=house_cities or ["karachi"])
            province_name = st.selectbox("Province", options=provinces or ["sindh"])
            purpose = st.selectbox("Purpose", options=purposes or ["for sale"])

            predict_house = st.form_submit_button("Predict House Price")

        if predict_house:
            x_house = build_house_features(
                house_metrics,
                {
                    "Total_Area": total_area,
                    "bedrooms": bedrooms,
                    "baths": baths,
                    "latitude": latitude,
                    "longitude": longitude,
                    "listing_year": listing_year,
                    "listing_month": listing_month,
                    "property_type": property_type,
                    "location": location,
                    "city": city,
                    "province_name": province_name,
                    "purpose": purpose,
                },
            )
            predicted = float(house_model.predict(x_house)[0])
            st.success(f"Predicted house price: {fmt_pkr(predicted)}")

    st.markdown("<div id='ai-assistant'></div>", unsafe_allow_html=True)
    st.subheader("AI Assistant")

    advisor_prompt = load_advisor_prompt_from_file()
    search_car_listings, search_house_listings, tools_error = load_advisor_tools()

    with st.expander("Advisor role instructions (loaded from advisor.py if available)"):
        st.code(advisor_prompt, language="text")

    if tools_error:
        st.error(f"Could not initialize advisor tools from advisor.py stack: {tools_error}")
    else:
        text_tab, voice_tab = st.tabs(["Text", "Voice (near realtime)"])

        with text_tab:
            user_query = st.text_area(
                "Ask the assistant",
                placeholder="Example: Find car options under 3000000 in lahore.\nExample: House options under 20000000 in islamabad.",
                height=100,
            )
            if st.button("Get Advisor Response"):
                response = assistant_answer(
                    user_query,
                    car_cities=car_cities,
                    house_cities=house_cities,
                    search_car_listings=search_car_listings,
                    search_house_listings=search_house_listings,
                )
                st.info(response)

        with voice_tab:
            st.caption("Speak a query, transcribe it, then get a spoken response in your browser.")
            speak_response = st.toggle("Speak assistant response", value=True)
            audio_value = st.audio_input("Voice input")

            if st.button("Transcribe and Ask"):
                if not audio_value:
                    st.warning("Please record a voice query first.")
                else:
                    transcript, transcript_error = transcribe_audio_with_openai(audio_value.getvalue())
                    if transcript_error:
                        st.error(transcript_error)
                    else:
                        st.write(f"**You said:** {transcript}")
                        response = assistant_answer(
                            transcript or "",
                            car_cities=car_cities,
                            house_cities=house_cities,
                            search_car_listings=search_car_listings,
                            search_house_listings=search_house_listings,
                        )
                        st.info(response)
                        if speak_response:
                            speak_text_in_browser(response)


if __name__ == "__main__":
    main()



