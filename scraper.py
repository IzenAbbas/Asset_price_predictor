"""Webpage text scraper for car listing pages.

This is the script version of the notebook scraper. It fetches a URL,
extracts visible text with requests + BeautifulSoup, removes boilerplate,
and saves the cleaned text to scraped_output.txt by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def scrape_page_text(url: str, timeout: int = 10) -> str:
    """Fetch a page and return the cleaned visible text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(["script", "style", "noscript", "meta", "link"]):
        element.decompose()

    raw_text = soup.get_text(separator="\n", strip=True)
    cleaned_lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    return "\n".join(cleaned_lines)


def save_text_to_file(text: str, path: str = "scraped_output.txt") -> None:
    """Save the provided text to a UTF-8 encoded file."""
    output_path = Path(path)
    output_path.write_text(text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape a webpage and save cleaned visible text.")
    parser.add_argument("url", help="Target webpage URL")
    parser.add_argument(
        "--output",
        default="scraped_output.txt",
        help="Output file path for the scraped text (default: scraped_output.txt)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        print(f"Fetching: {args.url}")
        text = scrape_page_text(args.url)
        save_text_to_file(text, args.output)
        print(f"Saved output to {args.output}")
        return 0
    except requests.exceptions.HTTPError as exc:
        print(f"HTTP error: {exc}")
        return 3
    except requests.exceptions.RequestException as exc:
        print(f"Network error: {exc}")
        return 4
    except Exception as exc:
        print(f"Error: {exc}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())