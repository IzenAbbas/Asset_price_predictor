from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from scraper import scrape_page_text as scraper_scrape_page_text


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
GEMINI_MODEL = "gemini-1.5-flash"
TARGET_FIELDS = (
	"model_year",
	"mileage_km",
	"engine_capacity_cc",
	"fuel_type",
	"transmission",
	"assembly",
	"brand",
	"model_name",
)

NOISE_PHRASES = {
	"download app",
	"sign up",
	"sign in",
	"home",
	"vehicles",
	"cars",
	"used cars",
	"search",
	"popular categories",
	"popular brands",
	"popular cities",
	"similar ads",
	"featured",
	"seller details",
	"safety tips",
	"report this ad",
	"post an ad",
	"notify me",
	"download our app now",
	"learn more",
	"continue",
	"view +",
	"view more",
	"ad id",
	"car info",
	"car details",
	"seller's comments",
	"details",
	"features",
	"description",
}

SIGNAL_KEYWORDS = (
	"year",
	"model",
	"mileage",
	"km",
	"engine",
	"cc",
	"fuel",
	"transmission",
	"assembly",
	"make",
	"brand",
	"variant",
	"description",
	"petrol",
	"diesel",
	"automatic",
	"manual",
	"imported",
	"local",
)


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


def scrape_page_text(url: str) -> tuple[str, str]:
	headers = {
		"User-Agent": (
			"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
			"AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
		),
		"Accept-Language": "en-US,en;q=0.9",
	}
	response = requests.get(url, headers=headers, timeout=30)
	response.raise_for_status()

	soup = BeautifulSoup(response.text, "html.parser")
	for tag in soup(["script", "style", "noscript", "svg", "iframe", "form", "button"]):
		tag.decompose()

	title = normalize_spaces(soup.title.get_text(" ", strip=True)) if soup.title else ""
	raw_text = soup.get_text("\n", strip=True)
	return title, raw_text


def is_noise_line(line: str) -> bool:
	lower = line.lower()
	if not line:
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
	if re.search(r"\b(19|20)\d{2}\b", line):
		return True
	if re.search(r"\b\d[\d,]*\s?(?:km|cc)\b", lower):
		return True
	if re.search(r"\b(?:petrol|diesel|hybrid|electric|automatic|manual|imported|local)\b", lower):
		return True
	if len(line) <= 90 and re.search(r"[A-Za-z]", line) and re.search(r"\d", line):
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


def clean_pakwheels(text: str) -> str:
	# First: try to extract the actual ad block from the raw text before doing global removals.
	raw_lines = [ln for ln in text.splitlines()]
	for i, line in enumerate(raw_lines):
		if re.search(r"\b(19\d{2}|20\d{2})\b", line):
			start = max(0, i - 6)
			end = min(len(raw_lines), i + 12)
			window = "\n".join(raw_lines[start:end])
			# If core specs like km or cc are present, try to include them. Otherwise, look for spec lines elsewhere and expand.
			if re.search(r"\b\d[\d,]*\s?(?:km|kms|kilometres?|kilometers?)\b", window, flags=re.IGNORECASE) or re.search(r"\b\d{3,4}\s?cc\b", window, flags=re.IGNORECASE):
				# ensure we also include nearby spec blocks such as Fuel/Transmission/Assembly if they appear elsewhere
				spec_match = re.search(r"\b(petrol|diesel|hybrid|electric|automatic|manual|assembly|engine capacity)\b", window, flags=re.IGNORECASE)
				if spec_match:
					return window.strip()
				# otherwise search the whole document for spec-like lines and expand window to include them
				for j, l in enumerate(raw_lines):
					if re.search(r"\b(petrol|diesel|hybrid|electric|automatic|manual|assembly|engine capacity)\b", l, flags=re.IGNORECASE):
						# expand to include this spec block
						start = min(start, max(0, j - 3))
						end = max(end, min(len(raw_lines), j + 4))
						return "\n".join(raw_lines[start:end]).strip()

	# If ad block not found, fall back to removing top navigation, service promos and long SEO/link lists
	cleaned = re.sub(r"^(Used Cars|New Cars|Bikes|Auto Store|Videos|Forums|Blog|PakWheels.*|Auction Sheet.*|MTMIS.*|DLIMS.*).*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
	cleaned = re.sub(r"(?s)(SIMILAR ADS|Used [A-Za-z ]+ by Year|Used [A-Za-z ]+ by City|Notify Me).*$", "", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"^(Send Message|Show Phone Number|Learn More|×|Previous|Next)$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)
	cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
	return cleaned.strip()


def clean_olx(text: str) -> str:
	# Remove everything before the breadcrumbs: Home\nVehicles\nCars
	text = re.sub(r"(?s)^.*?(?=Home\nVehicles\nCars)", "", text, flags=re.IGNORECASE)
	# Remove safety tips and app promos and inspection banners
	text = re.sub(r"(?s)(Your safety matters to us!|Find amazing deals on the go|Download OLX app now!).*$", "", text, flags=re.IGNORECASE)
	text = re.sub(r"^(OLX Car Inspection|Buy with confidence|200\+ checkup points|Book Now|Only meet in public).*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
	text = re.sub(r"^(Find amazing deals on the go|Download OLX app now!|Popular Categories|Trending Searches).*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
	text = re.sub(r"\n{2,}", "\n\n", text)
	return text.strip()


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

def extract_categorical_by_regex(cleaned_text: str) -> dict[str, Any]:
	"""Deterministic regex fallbacks for categorical fields.

	Attempts to extract fuel_type, transmission, assembly, brand and model_name
	from the cleaned listing text when the NER returns null.
	"""
	lower = cleaned_text.lower()
	lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
	res: dict[str, Any] = {"fuel_type": None, "transmission": None, "assembly": None, "brand": None, "model_name": None}

	# Fuel type: prefer whole-line tokens or labeled 'fuel' lines
	for line in lines:
		ll = line.lower()
		exact_fuel = re.fullmatch(r"\s*(petrol|diesel|hybrid|electric)\s*", ll, flags=re.IGNORECASE)
		if exact_fuel:
			res["fuel_type"] = exact_fuel.group(1).lower()
			break
		if "fuel" in ll:
			m = re.search(r"\b(petrol|diesel|hybrid|electric)\b", ll, flags=re.IGNORECASE)
			if m:
				res["fuel_type"] = m.group(1).lower()
				break

	# last resort: anywhere in text
	if res["fuel_type"] is None:
		m_any = re.search(r"\b(petrol|diesel|hybrid|electric)\b", lower, flags=re.IGNORECASE)
		if m_any:
			res["fuel_type"] = m_any.group(1).lower()

	# Transmission: prefer exact 'automatic' or 'manual'; map common variants
	trans_exact = None
	for line in lines:
		m = re.search(r"\b(automatic|manual)\b", line, flags=re.IGNORECASE)
		if m:
			trans_exact = m.group(1).lower()
			break
	if trans_exact:
		res["transmission"] = trans_exact
	else:
		variant_map = {
			"prosmatec": "automatic",
			"cvt": "automatic",
			"semi-automatic": "automatic",
			"semi automatic": "automatic",
			"auto": "automatic",
			"at": "automatic",
			"mt": "manual",
		}
		for v, mapped in variant_map.items():
			if re.search(r"\b" + re.escape(v) + r"\b", lower, flags=re.IGNORECASE):
				res["transmission"] = mapped
				break

	# Assembly
	asm_match = re.search(r"\b(local|imported|import)\b", lower, flags=re.IGNORECASE)
	if asm_match:
		asm = asm_match.group(1).lower()
		if asm == "import":
			asm = "imported"
		res["assembly"] = asm

	# Brand and model heuristics
	COMMON_BRANDS = [
		"Toyota",
		"Honda",
		"Suzuki",
		"Kia",
		"Hyundai",
		"Changan",
		"MG",
		"BMW",
		"Audi",
		"Daihatsu",
		"Nissan",
	]


	brand_found = None
	model_found = None
	CITY_TOKENS = ["Rawalpindi", "Karachi", "Lahore", "Islamabad", "Peshawar", "Multan", "Faisalabad", "Gujranwala", "Sialkot", "Sargodha", "Bahawalpur"]

	# Prefer brand+model lines where model is not a city token. Search by brand first to prioritize correct brand detection.
	for b in COMMON_BRANDS:
		for line in lines:
			if re.search(r"\b" + re.escape(b) + r"\b", line, flags=re.IGNORECASE):
				# try to capture a model name that appears after the brand and before a year/city/km/price
				m = re.search(
					r"\b" + re.escape(b) + r"\b\s+(?P<model>[A-Za-z0-9 \-\/]+?)(?:\s+\b(19\d{2}|20\d{2})\b|\s+\b(?:" + "|".join(CITY_TOKENS) + r")\b|\s+\d[\d,]*\s?km\b|\s+pkR\b|$)",
					line,
					flags=re.IGNORECASE,
				)
				if m:
					cand = m.group("model").strip()
					cand = re.sub(r"\b(automatic|petrol|diesel|cc|km|lacs?)\b.*$", "", cand, flags=re.IGNORECASE).strip()
					cand = re.sub(r"[\|\,]+$", "", cand).strip()
					# if candidate looks like a city, skip it
					if any(re.search(r"\b" + re.escape(c) + r"\b", cand, flags=re.IGNORECASE) for c in CITY_TOKENS):
						continue
					brand_found = b.lower()
					model_found = cand
					break
		if brand_found:
			break

	# If not found in first lines, search entire text for a brand occurrence and nearby tokens
	if not brand_found:
		for b in COMMON_BRANDS:
			m = re.search(r"\b" + re.escape(b) + r"\b(.{0,40})\b(\w+)?", cleaned_text, flags=re.IGNORECASE)
			if m:
				brand_found = b.lower()
				if m.group(2):
					model_found = m.group(2).strip()
					break

	# As a fallback, try to parse the title-like first long line
	if not model_found and lines:
		first = lines[0]
		m = re.search(r"(?:Used\s+)?(?P<title>.+?)\s+\b(19\d{2}|20\d{2})\b", first, flags=re.IGNORECASE)
		if m:
			t = m.group("title")
			# attempt to split brand + model
			for b in COMMON_BRANDS:
				if re.search(r"\b" + re.escape(b) + r"\b", t, flags=re.IGNORECASE):
					brand_found = b.lower()
					model_found = re.sub(r"\b" + re.escape(b) + r"\b", "", t, flags=re.IGNORECASE).strip()
					# if model looks like a city, skip
					if any(re.search(r"\b" + re.escape(c) + r"\b", model_found, flags=re.IGNORECASE) for c in CITY_TOKENS):
						model_found = None
					break

	if brand_found:
		res["brand"] = brand_found
	if model_found:
		# normalize model by removing year tokens and stray separators
		model_found = re.sub(r"\b(19\d{2}|20\d{2})\b", "", model_found)
		model_found = re.sub(r"[\|\-\/]+", " ", model_found).strip()
		# if the extracted model is a known city, drop it
		if any(re.search(r"\b" + re.escape(c) + r"\b", model_found, flags=re.IGNORECASE) for c in CITY_TOKENS):
			model_found = None
		else:
			res["model_name"] = model_found.lower()

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
			if field in {"model_year", "mileage_km", "engine_capacity_cc"}:
				numeric_value = re.sub(r"[^\d]", "", value)
				if numeric_value:
					value = int(numeric_value)
		result[field] = value

	if result["model_year"] is None:
		model_year = find_number_near_labels(
			lines,
			(r"\bmodel year\b", r"\byear\b", r"\bmodel\b"),
			r"\b(19\d{2}|20\d{2})\b",
		)
		if model_year is not None and 1980 <= model_year <= 2035:
			result["model_year"] = model_year
		else:
			year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", cleaned_text)
			for year in year_matches:
				year_int = int(year)
				if 1980 <= year_int <= 2035:
					result["model_year"] = year_int
					break

	if result["mileage_km"] is None:
		mileage_km = find_number_near_labels(
			lines,
			(r"\bmileage\b", r"\bkm['\s]?s driven\b", r"\bkm driven\b", r"\bkilometers?\b", r"\bkilometres?\b"),
			r"\b(\d[\d,]*)\s?(?:km|kms|kilometers?|kilometres?)\b",
		)
		if mileage_km is not None:
			result["mileage_km"] = mileage_km
		else:
			mileage_match = re.search(r"\b(\d[\d,]*)\s?(?:km|kms|kilometers?|kilometres?)\b", cleaned_text, flags=re.IGNORECASE)
			if mileage_match:
				result["mileage_km"] = int(mileage_match.group(1).replace(",", ""))

	if result["engine_capacity_cc"] is None:
		engine_capacity = find_number_near_labels(
			lines,
			(r"\bengine\b", r"\bengine capacity\b"),
			r"\b(\d{3,4})\s?cc\b",
		)
		if engine_capacity is not None:
			result["engine_capacity_cc"] = engine_capacity
		else:
			engine_match = re.search(r"\b(\d{3,4})\s?cc\b", cleaned_text, flags=re.IGNORECASE)
			if engine_match:
				result["engine_capacity_cc"] = int(engine_match.group(1))

	for field in ("fuel_type", "transmission", "assembly", "brand", "model_name"):
		value = result[field]
		if isinstance(value, str):
			result[field] = value.lower()

	# If categorical fields are still missing, use deterministic regex fallbacks
	fallbacks = extract_categorical_by_regex(cleaned_text)
	for field in ("fuel_type", "transmission", "assembly", "brand", "model_name"):
		if result.get(field) is None and fallbacks.get(field):
			result[field] = fallbacks[field]

	return result


def call_google_ner(cleaned_text: str) -> dict[str, Any]:
	api_key = os.getenv("GOOGLE_API_KEY_NER_CARS") or os.getenv("GOOGLE_API_KEY")
	if not api_key:
		raise RuntimeError("Set GOOGLE_API_KEY_NER_CARS in your environment or .env file.")

	prompt = f"""
Extract structured vehicle fields from the cleaned car listing text below.

Return ONLY a JSON object with exactly these keys:
{json.dumps({field: None for field in TARGET_FIELDS}, indent=2)}

Rules:
- Use null when a value is not explicitly present.
- model_year, mileage_km, and engine_capacity_cc must be numbers or null.
- fuel_type, transmission, assembly, brand, and model_name should be lowercase strings or null.
- Do not guess.

Cleaned text:
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


def extract_vehicle_fields(url: str) -> dict[str, Any]:
	# Use the standalone scraper to fetch cleaned visible text
	raw_text = scraper_scrape_page_text(url)

	# Decide which platform-specific cleaner to use
	lower = url.lower()
	if "pakwheels" in lower:
		cleaned_text = clean_pakwheels(raw_text)
	elif "olx." in lower or "olx" in lower:
		cleaned_text = clean_olx(raw_text)
	else:
		raise RuntimeError("Unsupported domain; only PakWheels and OLX are supported.")

	# If the platform-specific cleaners didn't find much, fall back to heuristics
	if not cleaned_text.strip():
		cleaned_text = clean_listing_text(raw_text)

	try:
		return call_google_ner(cleaned_text)
	except Exception:
		return normalize_output({}, cleaned_text)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Scrape a car listing URL and extract key fields with Google NER.")
	parser.add_argument("url", nargs="?", help="Car listing URL to scrape. If omitted you'll be prompted.")
	return parser


def main() -> int:
	load_env_file()
	parser = build_arg_parser()
	args = parser.parse_args()

	url = args.url
	if not url:
		try:
			url = input("Enter PakWheels or OLX listing URL: ").strip()
		except EOFError:
			print("No URL provided.", file=sys.stderr)
			return 1

	lower = (url or "").lower()
	if not ("pakwheels" in lower or "olx." in lower or "olx" in lower):
		print("Error: only PakWheels or OLX URLs are supported.", file=sys.stderr)
		return 1

	try:
		result = extract_vehicle_fields(url)
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	print(json.dumps(result, indent=2, ensure_ascii=False))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
