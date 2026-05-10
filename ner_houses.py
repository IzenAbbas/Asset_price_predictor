from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests
from scraper import scrape_page_text as scraper_scrape_page_text


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
GEMINI_MODEL = "gemini-1.5-flash"
TARGET_FIELDS = (
	"Total_Area",
	"bedrooms",
	"baths",
	"latitude",
	"longitude",
	"listing_year",
	"listing_month",
	"property_type",
	"location",
	"city",
	"province_name",
	"purpose",
)

NOISE_PHRASES = {
	"properties",
	"plot finder",
	"area guides",
	"blog",
	"maps",
	"tools",
	"forum",
	"index",
	"trends",
	"add property",
	"change area unit",
	"change currency",
	"buy",
	"homes",
	"plots",
	"commercial",
	"rent",
	"agents",
	"new projects",
	"home loan",
	"share on facebook",
	"share on twitter",
	"share on whatsapp",
	"send via gmail",
	"send via e-mail",
	"overview",
	"location & nearby",
	"home finance",
	"price index",
	"safety tips for property transactions",
	"useful links",
	"popular searches",
	"our home partners",
	"company",
	"about us",
	"contact us",
	"jobs",
	"help & support",
	"advertise on zameen",
	"terms of use",
	"connect",
	"news",
	"expo",
	"real estate agents",
	"head office",
	"email us",
	"roshan digital account",
	"top",
}

SIGNAL_KEYWORDS = (
	"beds",
	"bedroom",
	"baths",
	"bathroom",
	"marla",
	"kanal",
	"sq ft",
	"square feet",
	"area",
	"price",
	"pkr",
	"purpose",
	"for sale",
	"for rent",
	"type",
	"house",
	"flat",
	"upper portion",
	"lower portion",
	"farm house",
	"penthouse",
	"room",
	"location",
	"city",
	"province",
	"latitude",
	"longitude",
	"added",
)

PROPERTY_TYPES = {
	"house",
	"flat",
	"upper portion",
	"lower portion",
	"farm house",
	"penthouse",
	"room",
	"residential plot",
	"commercial",
}

MONTH_MAP = {
	"jan": 1,
	"january": 1,
	"feb": 2,
	"february": 2,
	"mar": 3,
	"march": 3,
	"apr": 4,
	"april": 4,
	"may": 5,
	"jun": 6,
	"june": 6,
	"jul": 7,
	"july": 7,
	"aug": 8,
	"august": 8,
	"sep": 9,
	"september": 9,
	"oct": 10,
	"october": 10,
	"nov": 11,
	"november": 11,
	"dec": 12,
	"december": 12,
}


def load_env_file() -> None:
	if not ENV_FILE.exists():
		return

	for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip()
		if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
			value = value[1:-1]
		os.environ.setdefault(key, value)


def normalize_spaces(text: str) -> str:
	return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def is_noise_line(line: str) -> bool:
	lower = line.lower().strip()
	if not lower:
		return True
	if lower in NOISE_PHRASES:
		return True
	if any(lower.startswith(phrase) for phrase in NOISE_PHRASES):
		return True
	if re.fullmatch(r"[\W_]+", line):
		return True
	if re.fullmatch(r"\d{1,3}", line):
		return True
	if len(line) <= 2:
		return True
	return False


def is_signal_line(line: str) -> bool:
	lower = line.lower()
	if any(keyword in lower for keyword in SIGNAL_KEYWORDS):
		return True
	if re.search(r"\b\d[\d,.]*\s?(?:marla|kanal|sq\s?ft|sq\s?feet|square feet)\b", lower):
		return True
	if re.search(r"\b\d+\s?(?:beds?|bedrooms?|baths?|bathrooms?)\b", lower):
		return True
	if re.search(r"\b(?:pk[r]?\b|crore|lakh)\b", lower):
		return True
	if re.search(r"\b(19|20)\d{2}\b", lower):
		return True
	if re.search(r"\b-?\d{1,2}\.\d{3,6}\b", lower):
		return True
	return False


def clean_listing_text(raw_text: str, title: str = "") -> str:
	candidates: list[str] = []
	seen: set[str] = set()

	if title:
		title_line = normalize_spaces(title)
		if title_line and title_line.lower() not in seen:
			candidates.append(title_line)
			seen.add(title_line.lower())

	for raw_line in raw_text.splitlines():
		line = normalize_spaces(raw_line)
		if is_noise_line(line):
			continue
		if not is_signal_line(line):
			continue
		lowered = line.lower()
		if lowered not in seen:
			candidates.append(line)
			seen.add(lowered)

	if not candidates:
		return normalize_spaces(raw_text)

	return "\n".join(candidates)


def _extract_structured_pairs(lines: list[str]) -> list[str]:
	labels = {
		"type",
		"price",
		"bath(s)",
		"baths",
		"area",
		"purpose",
		"bedroom(s)",
		"bedrooms",
		"added",
		"location",
		"latitude",
		"longitude",
	}
	pairs: list[str] = []
	i = 0
	while i < len(lines):
		line = lines[i].strip()
		lower = line.lower()
		if lower in labels and i + 1 < len(lines):
			val = lines[i + 1].strip()
			if val:
				pairs.append(line)
				pairs.append(val)
				i += 2
				continue
		i += 1
	return pairs


def clean_zameen(text: str) -> str:
	raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

	title_line = ""
	for line in raw_lines:
		if " for sale " in line.lower() or " for rent " in line.lower():
			if len(line) > len(title_line) and "zameen" not in line.lower():
				title_line = line

	structured_pairs = _extract_structured_pairs(raw_lines)

	signal_lines: list[str] = []
	for line in raw_lines:
		if is_noise_line(line):
			continue
		if is_signal_line(line):
			signal_lines.append(line)

	parts: list[str] = []
	if title_line:
		parts.append(f"Title: {title_line}")
	if structured_pairs:
		parts.append("Specs:\n" + "\n".join(structured_pairs))
	if signal_lines:
		parts.append("Signals:\n" + "\n".join(signal_lines[:40]))

	if parts:
		return "\n\n".join(parts)

	return clean_listing_text(text)


def extract_json_block(text: str) -> dict[str, Any]:
	cleaned = text.strip()
	cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"\s*```$", "", cleaned)

	try:
		return json.loads(cleaned)
	except json.JSONDecodeError:
		match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
		if match:
			return json.loads(match.group(0))
		raise


def _parse_total_area(text: str) -> float | None:
	match = re.search(
		r"\b(\d[\d,.]*)\s?(marla|kanal|sq\s?ft|sq\s?feet|square feet)\b",
		text,
		flags=re.IGNORECASE,
	)
	if not match:
		return None
	numeric = match.group(1).replace(",", "")
	unit = match.group(2).lower()
	try:
		value = float(numeric)
	except ValueError:
		return None

	if unit == "marla":
		return value * 272.25
	if unit == "kanal":
		return value * 20 * 272.25
	return value


def _parse_month(text: str) -> int | None:
	lower = text.lower()
	for key, value in MONTH_MAP.items():
		if re.search(r"\b" + re.escape(key) + r"\b", lower):
			return value
	return None


def find_number_near_labels(lines: list[str], label_patterns: tuple[str, ...], value_pattern: str) -> int | None:
	for index, line in enumerate(lines):
		lower = line.lower()
		if not any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in label_patterns):
			continue

		window = lines[index : min(len(lines), index + 3)]
		for candidate in window:
			match = re.search(value_pattern, candidate, flags=re.IGNORECASE)
			if match:
				return int(match.group(1).replace(",", ""))
	return None


def extract_categorical_by_regex(cleaned_text: str) -> dict[str, Any]:
	lower = cleaned_text.lower()
	lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
	res: dict[str, Any] = {
		"property_type": None,
		"purpose": None,
		"location": None,
		"city": None,
		"province_name": None,
	}

	for prop in PROPERTY_TYPES:
		if re.search(r"\b" + re.escape(prop) + r"\b", lower):
			res["property_type"] = prop
			break

	if re.search(r"\bfor sale\b", lower):
		res["purpose"] = "for sale"
	elif re.search(r"\bfor rent\b", lower):
		res["purpose"] = "for rent"

	# Parse location line like "DHA Phase 6, DHA Defence, Lahore, Punjab"
	for line in lines:
		if "," in line and any(city in line.lower() for city in ("lahore", "karachi", "islamabad", "rawalpindi", "peshawar", "multan", "quetta")):
			parts = [p.strip() for p in line.split(",") if p.strip()]
			if len(parts) >= 2:
				res["province_name"] = parts[-1].lower()
				res["city"] = parts[-2].lower()
				res["location"] = parts[0].lower()
				break

	return res


def normalize_output(data: dict[str, Any], cleaned_text: str) -> dict[str, Any]:
	result: dict[str, Any] = {field: None for field in TARGET_FIELDS}
	lines = [normalize_spaces(line) for line in cleaned_text.splitlines() if normalize_spaces(line)]

	for field in TARGET_FIELDS:
		value = data.get(field)
		if value is None:
			continue
		if isinstance(value, str):
			value = normalize_spaces(value)
			if not value or value.lower() in {"null", "none", "na", "n/a"}:
				continue
			if field in {"bedrooms", "baths", "listing_year", "listing_month"}:
				numeric_value = re.sub(r"[^\d]", "", value)
				if numeric_value:
					value = int(numeric_value)
			if field == "Total_Area":
				parsed = _parse_total_area(value)
				value = parsed if parsed is not None else value
		elif field == "Total_Area" and isinstance(value, (int, float)):
			if re.search(r"\bmarla\b", cleaned_text, flags=re.IGNORECASE):
				value = float(value) * 272.25
			elif re.search(r"\bkanal\b", cleaned_text, flags=re.IGNORECASE):
				value = float(value) * 20 * 272.25
		result[field] = value

	if result["Total_Area"] is None:
		area = _parse_total_area(cleaned_text)
		if area is not None:
			result["Total_Area"] = area

	if result["bedrooms"] is None:
		bedrooms = find_number_near_labels(
			lines,
			(r"\bbedroom\b", r"\bbedrooms\b", r"\bbeds?\b"),
			r"\b(\d{1,2})\b",
		)
		if bedrooms is not None:
			result["bedrooms"] = bedrooms

	if result["baths"] is None:
		baths = find_number_near_labels(
			lines,
			(r"\bbath\b", r"\bbaths\b", r"\bbathroom\b"),
			r"\b(\d{1,2})\b",
		)
		if baths is not None:
			result["baths"] = baths

	# Date parsing from lines like "Added May 10, 2024"
	if result["listing_year"] is None or result["listing_month"] is None:
		for line in lines:
			if "added" not in line.lower():
				continue
			year_match = re.search(r"\b(19|20)\d{2}\b", line)
			if year_match and result["listing_year"] is None:
				result["listing_year"] = int(year_match.group(0))
			if result["listing_month"] is None:
				month_val = _parse_month(line)
				if month_val:
					result["listing_month"] = month_val

	for field in ("property_type", "location", "city", "province_name", "purpose"):
		value = result[field]
		if isinstance(value, str):
			result[field] = value.lower()

	fallbacks = extract_categorical_by_regex(cleaned_text)
	for field in ("property_type", "location", "city", "province_name", "purpose"):
		if result.get(field) is None and fallbacks.get(field):
			result[field] = fallbacks[field]

	return result


def call_google_ner(cleaned_text: str) -> dict[str, Any]:
	api_key = os.getenv("GOOGLE_API_KEY_NER_HOUSES") or os.getenv("GOOGLE_API_KEY")
	if not api_key:
		raise RuntimeError("Set GOOGLE_API_KEY_NER_HOUSES in your environment or .env file.")

	prompt = f"""\
You are a property data extraction assistant. Extract structured fields from the Zameen listing text below.

Return ONLY a JSON object with exactly these keys:
{json.dumps({field: None for field in TARGET_FIELDS}, indent=2)}

Rules:
- ONLY extract values that are explicitly stated in the text. Never guess or infer.
- Total_Area must be a number (no units). If multiple areas exist, use the main listed area.
- bedrooms, baths, listing_year, listing_month must be integers or null.
- property_type examples: "house", "flat", "upper portion", "lower portion", "farm house", "penthouse", "room".
- purpose: "for sale" or "for rent" or null.
- location: neighborhood or phase (e.g. "dha phase 6").
- city: city name (e.g. "lahore").
- province_name: province/region name (e.g. "punjab").
- latitude/longitude: numeric values if explicitly present.
- Use null when a value is not present.

Example input:
  7 Marla House For Sale In DHA Phase 6 Lahore
  Type
  House
  Purpose
  For Sale
  Bedroom(s)
  3
  Bath(s)
  4
  Area
  7 Marla
  Location
  DHA Phase 6, DHA Defence, Lahore, Punjab

Example output:
  {{"Total_Area": 7, "bedrooms": 3, "baths": 4, "latitude": null, "longitude": null, "listing_year": null, "listing_month": null, "property_type": "house", "location": "dha phase 6", "city": "lahore", "province_name": "punjab", "purpose": "for sale"}}

Now extract from this listing:
{cleaned_text}
""".strip()

	endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
	payload = {
		"contents": [
			{
				"role": "user",
				"parts": [{"text": prompt}],
			},
		],
		"generationConfig": {
			"temperature": 0,
			"responseMimeType": "application/json",
		},
	}
	response = requests.post(endpoint, params={"key": api_key}, json=payload, timeout=60)
	response.raise_for_status()
	data = response.json()

	try:
		text = data["candidates"][0]["content"]["parts"][0]["text"]
	except (KeyError, IndexError, TypeError) as exc:
		raise RuntimeError(f"Unexpected Google NER response format: {data}") from exc

	parsed = extract_json_block(text)
	return normalize_output(parsed, cleaned_text)


def extract_house_fields(url: str) -> dict[str, Any]:
	raw_text = scraper_scrape_page_text(url)

	lower = url.lower()
	if "zameen.com" not in lower:
		raise RuntimeError("Unsupported domain; only Zameen URLs are supported.")

	cleaned_text = clean_zameen(raw_text)
	if not cleaned_text.strip():
		cleaned_text = clean_listing_text(raw_text)

	try:
		result = call_google_ner(cleaned_text)
	except Exception:
		result = normalize_output({}, cleaned_text)

	return result


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Scrape a Zameen listing URL and extract key fields with Google NER.")
	parser.add_argument("url", nargs="?", help="Zameen listing URL to scrape. If omitted you'll be prompted.")
	return parser


def main() -> int:
	load_env_file()
	parser = build_arg_parser()
	args = parser.parse_args()

	url = args.url
	if not url:
		try:
			url = input("Enter Zameen listing URL: ").strip()
		except EOFError:
			print("No URL provided.", file=sys.stderr)
			return 1

	lower = (url or "").lower()
	if "zameen.com" not in lower:
		print("Error: only Zameen URLs are supported.", file=sys.stderr)
		return 1

	try:
		result = extract_house_fields(url)
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	print(json.dumps(result, indent=2, ensure_ascii=False))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
