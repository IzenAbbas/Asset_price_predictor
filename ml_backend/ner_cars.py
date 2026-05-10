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


def _extract_structured_specs(raw_lines: list[str]) -> list[str]:
	"""Scan raw lines for structured label→value pairs used on PakWheels.

	PakWheels uses a pattern like:
	    Assembly        (label line)
	    Local           (value line)
	    Engine Capacity (label line)
	    1500 cc         (value line)

	This function detects those pairs and also collects standalone spec lines
	(e.g. "Petrol", "Automatic", "160,000 km").
	"""
	SPEC_LABELS = {
		"registered in", "color", "assembly", "engine capacity", "body type",
		"last updated", "fuel type", "fuel", "transmission", "mileage",
		"model year", "year", "make", "model", "variant", "engine type",
		"condition", "body color", "ad ref", "type", "registered city",
	}
	STANDALONE_SPEC_RE = re.compile(
		r"^(?:"
		r"\d[\d,]*\s?(?:km|kms|cc)\b"                         # 160,000 km  or  1500 cc
		r"|\.?\s*\d{3,4}\s?cc\s*\.?"                           # . 1500 cc .
		r"|\b(?:petrol|diesel|hybrid|electric|cng|lpg)\b"      # fuel types
		r"|\b(?:automatic|manual)\b"                           # transmission
		r"|\b(?:local|imported)\b"                             # assembly
		r"|\b(?:sedan|hatchback|suv|crossover|van|truck)\b"    # body type
		r"|\b(?:19|20)\d{2}\b"                                 # standalone year
		r")$",
		re.IGNORECASE,
	)

	specs: list[str] = []
	seen: set[str] = set()
	i = 0
	while i < len(raw_lines):
		line = raw_lines[i].strip()
		lower = line.lower()

		# Check if this line is a known spec label
		if lower in SPEC_LABELS:
			specs.append(line)
			seen.add(lower)
			# Grab the next non-empty line as the value
			if i + 1 < len(raw_lines):
				val = raw_lines[i + 1].strip()
				if val and val.lower() not in SPEC_LABELS:
					specs.append(val)
					seen.add(val.lower())
					i += 2
					continue
			i += 1
			continue

		# Check if this is a standalone spec value (e.g. "Petrol", "160,000 km")
		if STANDALONE_SPEC_RE.match(line) and lower not in seen:
			specs.append(line)
			seen.add(lower)

		i += 1

	return specs


def clean_pakwheels(text: str) -> str:
	raw_lines = [ln.strip() for ln in text.splitlines()]

	# --- Step 1: Extract the ad title ---
	# PakWheels titles appear as repeated heading lines like "Honda Civic EXi Prosmatec 2004"
	# They also appear in breadcrumbs. Look for a line with brand + year pattern.
	title_line = ""
	TITLE_RE = re.compile(
		r"^(?:Used\s+)?(?:(?:Toyota|Honda|Suzuki|Kia|Hyundai|Changan|MG|BMW|Audi|"
		r"Daihatsu|Nissan|Mercedes|Mitsubishi|Lexus|Mazda|Subaru|Isuzu|FAW|BAIC|"
		r"Prince|United|Chery|Proton|Peugeot|Volkswagen|Land Rover|Jeep)\s+)"
		r".+\b(?:19|20)\d{2}\b",
		re.IGNORECASE,
	)
	for line in raw_lines:
		if TITLE_RE.match(line):
			# Prefer the longest matching title (the actual ad title, not breadcrumb)
			if len(line) > len(title_line):
				title_line = line

	# --- Step 2: Collect quick-stats block ---
	# PakWheels shows a quick-stats section right after photos:
	#   2004
	#   160,000 km
	#   Petrol
	#   Automatic
	# Find this block: look for a standalone year followed by km/fuel/transmission lines
	quick_stats: list[str] = []
	for i, line in enumerate(raw_lines):
		if re.fullmatch(r"(19|20)\d{2}", line.strip()):
			# Check if the next few lines look like specs (km, fuel, transmission)
			window = raw_lines[i : min(len(raw_lines), i + 6)]
			has_km = any(re.search(r"\d[\d,]*\s?km", w, flags=re.IGNORECASE) for w in window)
			has_fuel = any(re.search(r"\b(petrol|diesel|hybrid|electric|cng|lpg)\b", w, flags=re.IGNORECASE) for w in window)
			has_price = any(re.search(r"\bpkr\b", w, flags=re.IGNORECASE) for w in window)
			# Require BOTH km and fuel, and NO price line (to skip the stats-bar near PKR)
			if has_km and has_fuel and not has_price:
				quick_stats = [w.strip() for w in window if w.strip() and not re.search(r"\bpkr\b|never buy|click photo|schedule|featured|previous|next", w, flags=re.IGNORECASE)]
				break

	# --- Step 3: Extract structured spec pairs (Assembly→Local, Engine Capacity→1500 cc) ---
	structured_specs = _extract_structured_specs(raw_lines)

	# --- Step 4: Extract seller's comments / description ---
	# PakWheels has "Seller's Comments" appearing twice: once in the tab nav
	# and once as the actual section header. We want the LAST occurrence.
	seller_comments: list[str] = []
	comment_start_idx = -1
	for i, line in enumerate(raw_lines):
		if line.lower().strip() in ("seller's comments", "seller comments"):
			comment_start_idx = i  # Keep updating to get the last occurrence
	if comment_start_idx >= 0:
		for line in raw_lines[comment_start_idx + 1:]:
			lower = line.lower().strip()
			# Stop at known section boundaries
			if lower in ("similar ads", "safety tips", "×", "reduce price", "seller details", ""):
				break
			if re.search(r"\b(mention pakwheels|pkr\s?\d)", lower, flags=re.IGNORECASE):
				break
			if len(line.strip()) > 3:
				seller_comments.append(line.strip())

	# --- Step 5: Assemble the clean text ---
	parts: list[str] = []
	if title_line:
		parts.append(f"Title: {title_line}")
	if quick_stats:
		parts.append("Quick Stats: " + " | ".join(quick_stats))
	if structured_specs:
		parts.append("Specs:\n" + "\n".join(structured_specs))
	if seller_comments:
		parts.append("Description: " + " ".join(seller_comments[:5]))

	if parts:
		return "\n\n".join(parts)

	# Fallback: remove navigation and SEO junk
	cleaned = re.sub(r"^(Used Cars|New Cars|Bikes|Auto Store|Videos|Forums|Blog|PakWheels.*|Auction Sheet.*|MTMIS.*|DLIMS.*).*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
	cleaned = re.sub(r"(?s)(SIMILAR ADS|Used [A-Za-z ]+ by Year|Used [A-Za-z ]+ by City|Notify Me).*$", "", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"^(Send Message|Show Phone Number|Learn More|×|Previous|Next)$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)
	cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
	return cleaned.strip()





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


def extract_brand_model_from_url(url: str) -> tuple[str | None, str | None]:
	"""Infer brand and model from common listing URL slugs.

	Example:
	- https://www.pakwheels.com/used-cars/honda-civic-2004-for-sale-in-rawalpindi-11465498
	  -> ("honda", "civic")
	"""
	lower_url = (url or "").lower()

	# PakWheels slug: /used-cars/<brand>-<model>-<year>-for-sale-...
	pk = re.search(r"/used-cars/([a-z0-9-]+)-for-sale", lower_url)
	if pk:
		slug = pk.group(1)
		tokens = [tok for tok in slug.split("-") if tok]
		if tokens:
			brand = tokens[0]
			model_tokens = [tok for tok in tokens[1:] if not re.fullmatch(r"(19|20)\d{2}", tok)]
			model_name = " ".join(model_tokens).strip() or None
			return brand or None, model_name

	# OLX style URLs can vary a lot; leave as None for now.
	return None, None

def extract_categorical_by_regex(cleaned_text: str) -> dict[str, Any]:
	"""Deterministic regex fallbacks for categorical fields.

	Attempts to extract fuel_type, transmission, assembly, brand and model_name
	from the cleaned listing text when the NER returns null.
	"""
	lower = cleaned_text.lower()
	lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
	res: dict[str, Any] = {"fuel_type": None, "transmission": None, "assembly": None, "brand": None, "model_name": None}

	# --- Fuel type ---
	# Check Quick Stats line first (highest signal)
	for line in lines:
		ll = line.lower()
		if ll.startswith("quick stats:"):
			m = re.search(r"\b(petrol|diesel|hybrid|electric|cng|lpg)\b", ll, flags=re.IGNORECASE)
			if m:
				res["fuel_type"] = m.group(1).lower()
				break
	# Then check standalone or labeled fuel lines
	if res["fuel_type"] is None:
		for line in lines:
			ll = line.lower()
			exact_fuel = re.fullmatch(r"\s*(petrol|diesel|hybrid|electric|cng|lpg)\s*", ll, flags=re.IGNORECASE)
			if exact_fuel:
				res["fuel_type"] = exact_fuel.group(1).lower()
				break
			if "fuel" in ll:
				m = re.search(r"\b(petrol|diesel|hybrid|electric|cng|lpg)\b", ll, flags=re.IGNORECASE)
				if m:
					res["fuel_type"] = m.group(1).lower()
					break
	# Last resort: anywhere
	if res["fuel_type"] is None:
		m_any = re.search(r"\b(petrol|diesel|hybrid|electric|cng|lpg)\b", lower, flags=re.IGNORECASE)
		if m_any:
			res["fuel_type"] = m_any.group(1).lower()

	# --- Transmission ---
	# Check Quick Stats line first
	for line in lines:
		ll = line.lower()
		if ll.startswith("quick stats:"):
			m = re.search(r"\b(automatic|manual)\b", ll, flags=re.IGNORECASE)
			if m:
				res["transmission"] = m.group(1).lower()
				break
	# Then check any line
	if res["transmission"] is None:
		for line in lines:
			m = re.search(r"\b(automatic|manual)\b", line, flags=re.IGNORECASE)
			if m:
				res["transmission"] = m.group(1).lower()
				break
	# Map common variant names
	if res["transmission"] is None:
		variant_map = {
			"prosmatec": "automatic",
			"prosmatic": "automatic",
			"cvt": "automatic",
			"tiptronic": "automatic",
			"dsg": "automatic",
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

	# --- Assembly ---
	# Prefer label→value pair
	for i, line in enumerate(lines):
		if line.lower().strip() == "assembly" and i + 1 < len(lines):
			val = lines[i + 1].lower().strip()
			if val in ("local", "imported"):
				res["assembly"] = val
				break
	if res["assembly"] is None:
		asm_match = re.search(r"\b(local|imported|import)\b", lower, flags=re.IGNORECASE)
		if asm_match:
			asm = asm_match.group(1).lower()
			if asm == "import":
				asm = "imported"
			res["assembly"] = asm

	# --- Brand and Model ---
	COMMON_BRANDS = [
		"Toyota", "Honda", "Suzuki", "Kia", "Hyundai", "Changan", "MG", "BMW",
		"Audi", "Daihatsu", "Nissan", "Mercedes", "Mitsubishi", "Lexus", "Mazda",
		"Subaru", "Isuzu", "FAW", "BAIC", "Prince", "United", "Chery", "Proton",
		"Peugeot", "Volkswagen", "Land Rover", "Jeep",
	]
	CITY_TOKENS = [
		"Rawalpindi", "Karachi", "Lahore", "Islamabad", "Peshawar", "Multan",
		"Faisalabad", "Gujranwala", "Sialkot", "Sargodha", "Bahawalpur",
		"Hyderabad", "Quetta", "Abbottabad", "Mardan",
	]
	# Noise tokens that should not be part of model_name
	NOISE_MODEL_TOKENS = {
		"cars", "car", "for", "sale", "in", "price", "pakistan", "used",
		"prices", "specs", "features", "variants", "review",
	}

	brand_found = None
	model_found = None

	# PRIORITY 1: Parse from "Title: Brand Model Variant Year" line (our structured output)
	for line in lines:
		if line.lower().startswith("title:"):
			title_text = line[6:].strip()  # Remove "Title: " prefix
			for b in COMMON_BRANDS:
				m = re.search(
					r"\b" + re.escape(b) + r"\b\s+(?P<model>[A-Za-z0-9][A-Za-z0-9 \-\/]*?)(?:\s+\b(?:19|20)\d{2}\b|$)",
					title_text,
					flags=re.IGNORECASE,
				)
				if m:
					brand_found = b.lower()
					cand = m.group("model").strip()
					# Extract just the core model name (first 1-2 words, strip variants like "EXi Prosmatec")
					model_parts = cand.split()
					if model_parts:
						# The first word is almost always the model name
						model_found = model_parts[0].lower()
					break
			break

	# PRIORITY 2: Search all lines for brand+model patterns
	if not brand_found:
		for b in COMMON_BRANDS:
			for line in lines:
				if re.search(r"\b" + re.escape(b) + r"\b", line, flags=re.IGNORECASE):
					m = re.search(
						r"\b" + re.escape(b) + r"\b\s+(?P<model>[A-Za-z0-9 \-\/]+?)(?:\s+\b(?:19|20)\d{2}\b|\s+\b(?:" + "|".join(CITY_TOKENS) + r")\b|\s+\d[\d,]*\s?km\b|\s+pkr\b|$)",
						line,
						flags=re.IGNORECASE,
					)
					if m:
						cand = m.group("model").strip()
						cand = re.sub(r"\b(automatic|manual|petrol|diesel|cc|km|lacs?)\b.*$", "", cand, flags=re.IGNORECASE).strip()
						cand = re.sub(r"[\|,]+$", "", cand).strip()
						# Reject city names
						if any(re.search(r"\b" + re.escape(c) + r"\b", cand, flags=re.IGNORECASE) for c in CITY_TOKENS):
							continue
						# Reject noise words
						cand_words = [w for w in cand.lower().split() if w not in NOISE_MODEL_TOKENS]
						if not cand_words:
							continue
						brand_found = b.lower()
						model_found = cand_words[0]  # Core model name
						break
			if brand_found:
				break

	if brand_found:
		res["brand"] = brand_found
	if model_found:
		model_found = re.sub(r"\b(19\d{2}|20\d{2})\b", "", model_found).strip()
		model_found = re.sub(r"[\|\-\/]+", " ", model_found).strip()
		if model_found and not any(re.search(r"\b" + re.escape(c) + r"\b", model_found, flags=re.IGNORECASE) for c in CITY_TOKENS):
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

	prompt = f"""\
You are a vehicle data extraction assistant. Extract structured fields from the car listing text below.

Return ONLY a JSON object with exactly these keys:
{json.dumps({field: None for field in TARGET_FIELDS}, indent=2)}

Rules:
- ONLY extract values that are explicitly stated in the text. Never guess or infer.
- model_year, mileage_km, engine_capacity_cc must be integers or null.
- mileage_km: extract the number in kilometres (e.g. "160,000 km" → 160000).
- engine_capacity_cc: extract the number in cc (e.g. "1500 cc" → 1500).
- fuel_type: one of "petrol", "diesel", "hybrid", "electric", "cng", "lpg", or null.
- transmission: "automatic" or "manual" or null. "Prosmatec" / "CVT" / "Tiptronic" = "automatic".
- assembly: "local" or "imported" or null.
- brand: the car manufacturer in lowercase (e.g. "honda", "toyota").
- model_name: the car model in lowercase (e.g. "civic", "corolla"). Do NOT include the brand, year, or variant.
- Use null when a value is not present.

Example input:
  Title: Honda Civic EXi Prosmatec 2004
  Quick Stats: 2004 | 160,000 km | Petrol | Automatic
  Specs:
  Assembly
  Local
  Engine Capacity
  1500 cc

Example output:
  {{"model_year": 2004, "mileage_km": 160000, "engine_capacity_cc": 1500, "fuel_type": "petrol", "transmission": "automatic", "assembly": "local", "brand": "honda", "model_name": "civic"}}

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


def extract_vehicle_fields(url: str) -> dict[str, Any]:
	# Use the standalone scraper to fetch cleaned visible text
	raw_text = scraper_scrape_page_text(url)

	lower = url.lower()
	if "pakwheels" not in lower:
		raise RuntimeError("Unsupported domain; only PakWheels URLs are supported.")

	cleaned_text = clean_pakwheels(raw_text)

	# If the platform-specific cleaner didn't find much, fall back to heuristics
	if not cleaned_text.strip():
		cleaned_text = clean_listing_text(raw_text)

	try:
		result = call_google_ner(cleaned_text)
	except Exception:
		result = normalize_output({}, cleaned_text)

	# Fallback: when NER misses brand/model, infer from URL slug.
	if result.get("brand") is None or result.get("model_name") is None:
		url_brand, url_model = extract_brand_model_from_url(url)
		if result.get("brand") is None and url_brand:
			result["brand"] = url_brand
		if result.get("model_name") is None and url_model:
			result["model_name"] = url_model

	return result


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
			url = input("Enter PakWheels listing URL: ").strip()
		except EOFError:
			print("No URL provided.", file=sys.stderr)
			return 1

	lower = (url or "").lower()
	if "pakwheels" not in lower:
		print("Error: only PakWheels URLs are supported.", file=sys.stderr)
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
