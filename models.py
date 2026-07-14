
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ── Media & Contacts ──────────────────────────────────────────────

class PropertyImage(BaseModel):
    """A single photo of a property."""
    url:     str
    caption: Optional[str] = None


class AgentInfo(BaseModel):
    """Contact details of the real-estate agent or broker."""
    name:    Optional[str] = None
    company: Optional[str] = None
    phone:   Optional[str] = None
    email:   Optional[str] = None


# ── Location ──────────────────────────────────────────────────────

class Location(BaseModel):
    """Full address + GPS coordinates of a property."""
    address:   Optional[str]   = None
    locality:  Optional[str]   = None   # neighbourhood / sector
    city:      Optional[str]   = None
    state:     Optional[str]   = None
    country:   Optional[str]   = "India"
    pincode:   Optional[str]   = None
    latitude:  Optional[float] = None
    longitude: Optional[float] = None


# ── Core Property Listing ─────────────────────────────────────────

class PropertyListing(BaseModel):
    """
    A single property listing scraped from any real-estate website.
    This is the central data object used throughout the entire pipeline.
    """
    # Identity
    source:      str   = Field(..., description="Website name e.g. magicbricks.com")
    url:         str
    property_id: Optional[str] = None
    title:       str
    description: Optional[str] = None

    # Property details
    property_type:    Optional[str]   = None   # Apartment / Villa / Plot
    listing_type:     Optional[str]   = "Sale" # Sale or Rent
    builder:          Optional[str]   = None
    price:            Optional[float] = None   # in INR
    currency:         str             = "INR"
    area_sqft:        Optional[float] = None
    bedrooms:         Optional[int]   = None
    bathrooms:        Optional[int]   = None
    balconies:        Optional[int]   = None
    floor:            Optional[str]   = None
    total_floors:     Optional[int]   = None
    furnished_status: Optional[str]   = None   # Furnished / Semi / Unfurnished
    availability:     Optional[str]   = None
    age_of_property:  Optional[str]   = None

    # Location & media
    location:    Location = Field(default_factory=Location)
    amenities:   List[str]           = Field(default_factory=list)
    images:      List[PropertyImage] = Field(default_factory=list)
    agent:       Optional[AgentInfo] = None
    posted_date: Optional[str]       = None
    scraped_at:  Optional[datetime] = Field(default_factory=datetime.utcnow)

    # AI scores — always reset to 0.0 on load, computed fresh each run
    semantic_score:   float         = 0.0
    rerank_score:     float         = 0.0
    investment_score: float         = 0.0
    travel_score:     float         = 0.0
    explanation:      Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _clean_nulls(cls, data: Any) -> Any:
        """Ensure lists aren't passed as None (null in JSON)."""
        if isinstance(data, dict):
            if data.get("amenities") is None:
                data["amenities"] = []
            if data.get("images") is None:
                data["images"] = []
            if data.get("location") is None:
                data["location"] = {}
        return data

    @model_validator(mode="after")
    def _set_scraped_at(self) -> "PropertyListing":
        """
        Auto-set scraped_at to the current UTC time if it was not provided
        or was explicitly set to None (common when ScrapeGraphAI returns
        null for internal fields it doesn't find on the page).
        """
        if self.scraped_at is None:
            self.scraped_at = datetime.utcnow()
        return self


# ── Search & Results ──────────────────────────────────────────────

class SearchQuery(BaseModel):
    """Structured version of the user's plain-English query."""
    user_query:          str
    city:                Optional[str]    = None
    locality:            Optional[str]    = None
    property_type:       Optional[str]    = None
    listing_type:        Optional[str]    = "Sale"
    bedrooms:            Optional[int]    = None
    min_budget:          Optional[float]  = None
    max_budget:          Optional[float]  = None
    must_have_amenities: List[str]        = Field(default_factory=list)
    nearby_preferences:  List[str]        = Field(default_factory=list)
    travel_preferences:  List[str]        = Field(default_factory=list)
    investment_goal:     Optional[str]    = None
    priority_weights:    Dict[str, float] = Field(default_factory=dict)
    compare_mode:        bool             = False
    explanation_style:   Optional[str]    = None


class SearchResult(BaseModel):
    """Final output of the search pipeline — a ranked list of properties."""
    query:         SearchQuery
    total_results: int
    properties:    List[PropertyListing]


class ComparisonResult(BaseModel):
    """Output when the user asks to compare multiple properties."""
    best_property:       PropertyListing
    compared_properties: List[PropertyListing]
    summary:             str


class ChatResponse(BaseModel):
    """What gets returned to app.py and displayed to the user."""
    answer:                 str
    sources:                List[str]
    recommended_properties: List[PropertyListing] = Field(default_factory=list)


class ScrapeBundle(BaseModel):
    """Summary stats returned after a scrape + store operation."""
    query:            str
    total_candidates: int
    stored_count:     int
    urls:             List[str] = Field(default_factory=list)


# ── Neighborhood ──────────────────────────────────────────────────

class NearbyPlace(BaseModel):
    """A single point-of-interest (school, hospital, metro etc.) near a property."""
    name:        str
    category:    str
    distance_km: float
    travel_time: Optional[str] = None


class NeighborhoodAnalysis(BaseModel):
    """All nearby places grouped by category for one property."""
    property_url: str
    schools:      List[NearbyPlace] = Field(default_factory=list)
    hospitals:    List[NearbyPlace] = Field(default_factory=list)
    metro:        List[NearbyPlace] = Field(default_factory=list)
    malls:        List[NearbyPlace] = Field(default_factory=list)
    restaurants:  List[NearbyPlace] = Field(default_factory=list)
    parks:        List[NearbyPlace] = Field(default_factory=list)


class MapPlace(BaseModel):
    """A place returned by the Google Maps Text Search API."""
    name:        str
    category:    str
    address:     Optional[str]   = None
    distance_km: Optional[float] = None
    travel_time: Optional[str]   = None


class NeighborhoodResult(BaseModel):
    """All places found for a neighborhood text query."""
    query:  str
    places: List[MapPlace] = Field(default_factory=list)
