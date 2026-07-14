
from __future__ import annotations

from typing import Dict, List

from scrapegraph_py import ScrapeGraphAI

from config import SCRAPEGRAPH_API_KEY
from models import PropertyListing


# ── Extraction Prompt ─────────────────────────────────────────────

EXTRACTION_PROMPT = """
Extract the complete property listing and return these fields:

- source website
- title
- property url
- property id
- property type (Apartment / Villa / Plot / etc.)
- listing type (Sale or Rent)
- builder

- price (number in INR)
- currency

- area in square feet

- bedrooms
- bathrooms
- balconies

- furnished status (Furnished / Semi-Furnished / Unfurnished)
- availability
- property age

- complete address
- locality
- city
- state
- pincode
- latitude
- longitude

- amenities (list)
- images (list of urls)
- agent information (name, company, phone, email)
- posted date
- description

Return NULL for any missing field.
""".strip()


# ── Scraper ───────────────────────────────────────────────────────

class PropertyScraper:
    """Scrapes full property details from a single listing URL."""

    def __init__(self) -> None:
        if not SCRAPEGRAPH_API_KEY:
            raise RuntimeError("SCRAPEGRAPH_API_KEY is not set in .env")
        self.client = ScrapeGraphAI(api_key=SCRAPEGRAPH_API_KEY)

    def scrape(self, url: str) -> PropertyListing:
        """
        Scrape one property URL and return a PropertyListing.
        Raises RuntimeError if the scrape fails.
        Raises ValidationError if the returned data is invalid.
        """
        response = self.client.extract(
            EXTRACTION_PROMPT,
            url=url,
            schema=PropertyListing.model_json_schema(),
        )

        if response.status != "success":
            raise RuntimeError(f"ScrapeGraphAI failed for {url}: {response.error}")

        return PropertyListing.model_validate(response.data.json_data)


# ── Search Service ────────────────────────────────────────────────

class PropertySearchService:
    """Finds property listing URLs for a plain-English search query."""

    def __init__(self) -> None:
        if not SCRAPEGRAPH_API_KEY:
            raise RuntimeError("SCRAPEGRAPH_API_KEY is not set in .env")
        self.client = ScrapeGraphAI(api_key=SCRAPEGRAPH_API_KEY)

    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Search for properties matching the query.
        Returns a list of {title, url, snippet} dicts.
        Raises RuntimeError if the search fails.
        """
        result = self.client.search(
            query=query,
            num_results=num_results,
            format="markdown",
        )

        if result.status != "success":
            raise RuntimeError(f"ScrapeGraphAI search failed: {result.error}")

        return [
            {
                "title":   item.title,
                "url":     item.url,
                "snippet": getattr(item, "snippet", ""),
            }
            for item in result.data.results
        ]


# ── Singletons ────────────────────────────────────────────────────

scraper        = PropertyScraper()
search_service = PropertySearchService()
