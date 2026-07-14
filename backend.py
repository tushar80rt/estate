
from __future__ import annotations

import json
import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from groq import Groq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient

from config import (
    APP_NAME,
    APP_VERSION,
    GOOGLE_API_KEY,
    GOOGLE_MAPS_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    MONGODB_CHUNK_COLLECTION,
    MONGODB_COLLECTION,
    MONGODB_DB,
    MONGODB_URI,
    NVIDIA_API_KEY,
)
from models import (
    AgentInfo,
    ChatResponse,
    ComparisonResult,
    Location,
    MapPlace,
    NearbyPlace,
    NeighborhoodAnalysis,
    NeighborhoodResult,
    PropertyImage,
    PropertyListing,
    ScrapeBundle,
    SearchQuery,
    SearchResult,
)
from scrapper import scraper, search_service

# SECTION 1 — LLM PROMPT TEMPLATES

ESTATEGPT_SYSTEM_PROMPT = """
You are EstateGPT, an agentic AI real-estate intelligence platform for India.

Your job is to help users search, compare, rank, and explain properties across multiple real-estate portals.
You must:
- understand natural-language property requirements
- search across sources using the available tools
- normalize listings into a unified structure
- remove duplicates
- rerank by semantic relevance to the user request
- enrich results with neighborhood intelligence
- compute practical investment and suitability scores
- explain recommendations with concise pros, cons, and tradeoffs

Always prefer grounded answers based on retrieved property data.
If data is missing, say so clearly instead of inventing it.
When asked for recommendations, return a ranked shortlist and explain why each property fits or fails the query.
""".strip()

STRUCTURE_QUERY_PROMPT = """
Convert the user request into JSON with these keys:
city, locality, property_type, listing_type, bedrooms, min_budget, max_budget,
must_have_amenities, nearby_preferences, travel_preferences, investment_goal,
priority_weights, compare_mode, explanation_style.

Rules:
- Return only valid JSON.
- Use null for missing scalar values.
- Use arrays for multi-value fields.
- Put monetary values in INR as numbers, not strings.
- Keep nearby_preferences short and concrete, such as metro, schools, hospitals, parks.
User request:
{query}
""".strip()

ANSWER_PROMPT = """
You are summarizing real-estate search results for the user request below.
Use only the provided properties and neighborhood notes.

User request:
{query}

Return:
1. A short recommendation summary.
2. A ranked list of the best properties.
3. A brief pros/cons paragraph for each recommended property.
4. A closing note that mentions any missing data or caveats.

Do not fabricate facts.
""".strip()

COMPARISON_PROMPT = """
Compare the selected properties for the user's request and explain which one fits best.
Focus on price, location, amenities, layout fit, investment quality, and commute convenience.
Do not invent details.
""".strip()


# SECTION 2 — MONGODB STORE
# All database reads and writes go through this class.


class MongoPropertyStore:
    """
    Wrapper around MongoDB with three collections:
      'properties'       → one document per property listing
      'property_chunks'  → text chunks + embeddings (powers RAG retrieval)
      'search_history'   → log of every user query (for analytics)
    """

    def __init__(self) -> None:
        # Lazy connection — only opens when first used
        self._client = None
        self._collection = None
        self._history_collection = None
        self._chunk_collection = None

    def _connect(self):
        """Open the MongoDB connection and create indexes (runs only once)."""
        if self._client is None:
            self._client = MongoClient(MONGODB_URI)
            db = self._client[MONGODB_DB]

            self._collection = db[MONGODB_COLLECTION]
            self._history_collection = db["search_history"]
            self._chunk_collection = db[MONGODB_CHUNK_COLLECTION]

            # Main properties collection indexes
            self._collection.create_index("url", unique=True)
            self._collection.create_index("source")
            self._collection.create_index(
                [("location.city", 1), ("location.locality", 1)]
            )
            self._collection.create_index(
                [
                    ("title", "text"),
                    ("description", "text"),
                    ("location.city", "text"),
                    ("location.locality", "text"),
                    ("amenities", "text"),
                ]
            )

            # Search history indexes
            self._history_collection.create_index("query")
            self._history_collection.create_index("created_at")

            # Chunks collection indexes
            self._chunk_collection.create_index(
                [("url", 1), ("chunk_index", 1)], unique=True
            )
            self._chunk_collection.create_index(
                [
                    ("chunk_text", "text"),
                    ("title", "text"),
                    ("city", "text"),
                    ("locality", "text"),
                ]
            )

        return self._collection

    # ── Search History ────────────────────────────────────────────

    def save_search_history(self, query: str) -> None:
        """Log the user's search query with a timestamp."""
        self._connect()
        if self._history_collection is None:
            return
        self._history_collection.insert_one(
            {
                "query": query,
                "created_at": datetime.utcnow().isoformat(),
            }
        )

    def get_search_history(self, limit: int = 15) -> list[str]:
        """Retrieve recent search queries."""
        self._connect()
        if self._history_collection is None:
            return []
        cursor = self._history_collection.find({}, {"query": 1}).sort("created_at", -1).limit(limit)
        
        # Deduplicate history while preserving order
        history = []
        seen = set()
        for doc in cursor:
            q = doc["query"]
            if q not in seen:
                seen.add(q)
                history.append(q)
        return history

    # ── Property Storage ──────────────────────────────────────────

    def upsert_many(self, listings: Sequence[PropertyListing]) -> int:
        """
        Insert or update multiple property listings.
        Also chunks + embeds each listing for RAG retrieval.
        Returns the number of listings saved/updated.
        """
        collection = self._connect()
        if collection is None:
            return 0

        saved_count = 0
        for listing in listings:
            payload = listing.model_dump()
            payload["updated_at"] = datetime.utcnow().isoformat()

            result = collection.update_one(
                {"url": listing.url},
                {"$set": payload},
                upsert=True,
            )
            self._upsert_chunks(listing)

            if (
                result.upserted_id is not None
                or result.modified_count
                or result.matched_count
            ):
                saved_count += 1

        return saved_count

    def _upsert_chunks(self, property_details: PropertyListing) -> None:
        """
        CHUNKING + EMBEDDING step of the RAG pipeline.
        Splits the listing into text chunks, embeds each with NVIDIAEmbeddings,
        and stores both in the 'property_chunks' collection.
        """
        if self._chunk_collection is None:
            self._connect()
        if self._chunk_collection is None:
            return

        # Delete old chunks before inserting fresh ones
        self._chunk_collection.delete_many({"url": property_details.url})

        full_description_text = _property_text(property_details)
        text_cutter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        small_text_pieces = text_cutter.split_text(full_description_text)

        for index_number, text_piece in enumerate(small_text_pieces):
            self._chunk_collection.insert_one(
                {
                    "url": property_details.url,
                    "source": property_details.source,
                    "title": property_details.title,
                    "city": property_details.location.city,
                    "locality": property_details.location.locality,
                    "chunk_index": index_number,
                    "chunk_text": text_piece,
                    "embedding": _embed_text(text_piece),
                    "created_at": datetime.utcnow().isoformat(),
                }
            )

    # ── Property Retrieval ────────────────────────────────────────

    def find_candidates(
        self,
        query: str,
        limit: int = 40,
        city: Optional[str] = None,
    ) -> List[PropertyListing]:
        """
        RETRIEVAL step — find properties matching the query via vector similarity.
        If `city` is provided, only chunks from that city are considered,
        preventing stale results from other cities leaking in.
        """
        collection = self._connect()
        if collection is None:
            return []

        # Score all chunks, keep the best score per property URL
        scored_urls: Dict[str, float] = {}
        for item in self._chunk_query(query, limit=limit * 5, city=city):
            url = item["url"]
            score = item["score"]
            if score > scored_urls.get(url, float("-inf")):
                scored_urls[url] = score

        if scored_urls:
            top_urls = [
                url
                for url, _ in sorted(
                    scored_urls.items(), key=lambda pair: pair[1], reverse=True
                )[:limit]
            ]
            # Also enforce city filter on the final property documents
            city_filter = (
                {
                    "url": {"$in": top_urls},
                    "location.city": {"$regex": city, "$options": "i"},
                }
                if city
                else {"url": {"$in": top_urls}}
            )
            docs = list(collection.find(city_filter))
            by_url = {doc.get("url", ""): normalize_listing(doc) for doc in docs}
            return [by_url[url] for url in top_urls if url in by_url]

        return []

    def _chunk_query(
        self,
        query: str,
        limit: int = 100,
        city: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search chunks by cosine similarity between query embedding and stored embeddings.
        If `city` is provided, adds a regex pre-filter on the city field.
        Returns [{url, score}] sorted by score descending.
        """
        if self._chunk_collection is None:
            self._connect()
        if self._chunk_collection is None:
            return []

        query_embedding = _embed_text(query)

        # Pre-filter by keyword + optional city to avoid scoring every chunk
        keywords = [w for w in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(w) > 2]
        mongo_query: Dict[str, Any] = {}
        if keywords:
            mongo_query = {
                "$or": (
                    [
                        {"chunk_text": {"$regex": w, "$options": "i"}}
                        for w in keywords[:8]
                    ]
                    + [{"title": {"$regex": w, "$options": "i"}} for w in keywords[:8]]
                )
            }
        # City filter: narrow chunks to the requested city only
        if city:
            mongo_query["city"] = {"$regex": city, "$options": "i"}

        docs = list(self._chunk_collection.find(mongo_query).limit(limit))

        scored: List[Dict[str, Any]] = []
        for doc in docs:
            score = _similarity_score(query_embedding, doc.get("embedding") or [])
            chunk_text = str(doc.get("chunk_text", "")).lower()
            for word in keywords[:8]:
                if word in chunk_text:
                    score += 0.03  # small bonus for exact keyword match
            scored.append({"url": doc.get("url", ""), "score": score})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored

    def all_urls(self) -> List[str]:
        """Return all stored property URLs (max 500)."""
        collection = self._connect()
        if collection is None:
            return []
        return [
            doc.get("url", "") for doc in collection.find({}, {"url": 1}).limit(500)
        ]


# Singleton — one shared MongoDB store used everywhere in this file
mongo_store = MongoPropertyStore()


# SECTION 3 — UTILITY HELPERS


def _normalize_text(value: Optional[str]) -> str:
    """Lowercase + collapse whitespace. Used for deduplication keys."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip().lower()


def _coerce_float(value: Any) -> Optional[float]:
    """Safely convert any value to float. Returns None on failure."""
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = re.sub(r"[^0-9.]", "", str(value))
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_int(value: Any) -> Optional[int]:
    """Safely convert any value to int. Returns None on failure."""
    numeric = _coerce_float(value)
    return int(numeric) if numeric is not None else None


def _json_loads(text: str) -> Dict[str, Any]:
    """Parse JSON that may be wrapped in markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def _infer_source(url: str) -> str:
    """Guess the website name from a URL (e.g. 'magicbricks.com')."""
    if not url:
        return "Unknown"
    host = re.sub(r"^https?://", "", url.lower()).split("/")[0]
    return host.replace("www.", "").split(":")[0]


# SECTION 4 — LLM CLIENT (Groq / LLaMA)


def _groq_client() -> Groq:
    """Return a Groq client. Raises RuntimeError if GROQ_API_KEY is not set."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    return Groq(api_key=GROQ_API_KEY)


def _model_text(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    """
    Send messages to Groq LLaMA and return the text response.
    Raises RuntimeError if GROQ_API_KEY is missing.
    """
    client = _groq_client()

    kwargs: Dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or ""


# SECTION 5 — LANGCHAIN TOOLS
# The 3 tools the LangGraph agent can call automatically.


@tool("estate_search")
def estate_search(query: str) -> str:
    """
    Tool 1 — Full property search pipeline.
    Scrapes property pages → stores in MongoDB → reranks → summarizes.
    Called when the user wants to FIND properties.
    """
    result = search_properties(query, num_results=10)
    ranked, notes = analyze_properties(query, result.properties)
    response = build_answer(query, ranked, notes)
    return response.model_dump_json(indent=2)


@tool("estate_compare")
def estate_compare(query: str) -> str:
    """
    Tool 2 — Property comparison.
    Retrieves top candidates, reranks, then asks LLM to pick the best.
    Called when the user says 'compare' or 'which is better'.
    """
    ranked = retrieve_and_rerank(query, limit=20)
    comparison = compare_properties(ranked[:5], query)
    response = ChatResponse(
        answer=comparison.summary,
        sources=[p.url for p in comparison.compared_properties],
        recommended_properties=list(comparison.compared_properties),
    )
    return response.model_dump_json(indent=2)


@tool("estate_neighborhood")
def estate_neighborhood(query: str) -> str:
    """
    Tool 3 — Neighborhood intelligence.
    Calls Google Maps to find nearby schools, hospitals, metro etc.
    Called when the user asks about the AREA around a property.
    """
    result = neighborhood_lookup(query)
    return result.model_dump_json(indent=2)


# SECTION 6 — QUERY PARSER
# Converts plain-English query → structured SearchQuery object.


def parse_user_query(user_query: str) -> SearchQuery:
    """
    Ask Groq LLM to extract all structured fields from the user's query.
    Raises ValueError if the LLM returns empty or unparseable output.
    """
    prompt = STRUCTURE_QUERY_PROMPT.format(query=user_query)
    text = _model_text(
        [
            {"role": "system", "content": ESTATEGPT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        json_mode=True,
    )
    if not text:
        raise ValueError(f"LLM returned empty response for query: {user_query}")

    data = _json_loads(text)
    return SearchQuery(
        user_query=user_query,
        city=data.get("city"),
        locality=data.get("locality"),
        property_type=data.get("property_type"),
        listing_type=data.get("listing_type") or "Sale",
        bedrooms=_coerce_int(data.get("bedrooms")),
        min_budget=_coerce_float(data.get("min_budget")),
        max_budget=_coerce_float(data.get("max_budget")),
        must_have_amenities=[
            str(i) for i in (data.get("must_have_amenities") or []) if i
        ],
        nearby_preferences=[
            str(i) for i in (data.get("nearby_preferences") or []) if i
        ],
        travel_preferences=[
            str(i) for i in (data.get("travel_preferences") or []) if i
        ],
        investment_goal=data.get("investment_goal"),
        priority_weights={
            k: float(v) for k, v in (data.get("priority_weights") or {}).items()
        },
        compare_mode=bool(data.get("compare_mode", False)),
        explanation_style=data.get("explanation_style"),
    )


# SECTION 7 — EMBEDDING & SIMILARITY  (RAG Core)


@lru_cache(maxsize=1)
def _embedding_model() -> NVIDIAEmbeddings:
    """Load the NVIDIAEmbeddings model once and cache it."""
    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5", nvidia_api_key=NVIDIA_API_KEY
    )


def _embed_text(text: str) -> List[float]:
    """Convert a text string into an embedding vector."""
    return _embedding_model().embed_query(text)


def _similarity_score(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Cosine similarity between two embedding vectors.
    Score range: -1.0 (opposite) to 1.0 (identical).
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    import numpy as np

    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _score_semantic(query: str, text: str) -> float:
    """Fast semantic similarity using NVIDIA embeddings."""
    # Truncate both to safely fit within NVIDIA's 512 token limit (~2000 chars)
    # This guarantees we will NEVER throw a 400 token limit error.
    q_vec = _embed_text(query[:1800])
    t_vec = _embed_text(text[:1800])
    return _similarity_score(q_vec, t_vec)


# SECTION 8 — DATA HELPERS
# Normalization, deduplication, and text building.


def _property_text(listing: PropertyListing) -> str:
    """Combine all important fields into a single string for embedding."""
    return " | ".join(
        part
        for part in [
            listing.title,
            listing.description or "",
            listing.property_type or "",
            listing.listing_type or "",
            listing.builder or "",
            listing.location.address or "",
            listing.location.locality or "",
            listing.location.city or "",
            ", ".join(listing.amenities),
        ]
        if part
    )


def normalize_listing(raw: Dict[str, Any] | PropertyListing) -> PropertyListing:
    """
    Convert a raw dict (from scraper or MongoDB) into a typed PropertyListing.
    Always resets score fields to 0.0 so they are computed fresh.

    Note: scrapper.py imports PropertyListing from models.py. To avoid any
    class-identity mismatch, we always convert Pydantic objects to dict first.
    """
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()

    if isinstance(raw, dict):
        for key in (
            "scraped_at",
            "semantic_score",
            "rerank_score",
            "investment_score",
            "travel_score",
            "explanation",
        ):
            raw.pop(key, None)

    if not isinstance(raw, dict):
        raise TypeError(
            f"normalize_listing: expected dict or Pydantic model, got {type(raw)}"
        )

    location = raw.get("location") or {}
    if hasattr(location, "model_dump"):
        location = location.model_dump()
    elif not isinstance(location, dict):
        location = {}

    listing = PropertyListing(
        source=str(raw.get("source") or _infer_source(str(raw.get("url") or ""))),
        url=str(raw.get("url") or raw.get("property_url") or ""),
        property_id=raw.get("property_id"),
        title=str(raw.get("title") or "Untitled listing"),
        description=raw.get("description"),
        property_type=raw.get("property_type"),
        listing_type=raw.get("listing_type") or "Sale",
        builder=raw.get("builder"),
        price=_coerce_float(raw.get("price")),
        currency=str(raw.get("currency") or "INR"),
        area_sqft=_coerce_float(raw.get("area_sqft")),
        bedrooms=_coerce_int(raw.get("bedrooms")),
        bathrooms=_coerce_int(raw.get("bathrooms")),
        balconies=_coerce_int(raw.get("balconies")),
        floor=raw.get("floor"),
        total_floors=_coerce_int(raw.get("total_floors")),
        furnished_status=raw.get("furnished_status"),
        availability=raw.get("availability"),
        age_of_property=raw.get("age_of_property"),
        location=Location(
            address=location.get("address"),
            locality=location.get("locality"),
            city=location.get("city"),
            state=location.get("state"),
            country=location.get("country") or "India",
            pincode=location.get("pincode"),
            latitude=_coerce_float(location.get("latitude")),
            longitude=_coerce_float(location.get("longitude")),
        ),
        amenities=[str(i) for i in (raw.get("amenities") or []) if i],
        images=[
            (
                PropertyImage.model_validate(i)
                if isinstance(i, dict)
                else PropertyImage(url=str(i))
            )
            for i in (raw.get("images") or [])
            if i
        ],
        agent=AgentInfo.model_validate(raw.get("agent")) if raw.get("agent") else None,
        posted_date=raw.get("posted_date"),
    )

    # Always reset scores — they are computed fresh in this session
    listing.semantic_score = 0.0
    listing.rerank_score = 0.0
    listing.investment_score = 0.0
    listing.travel_score = 0.0
    return listing


def deduplicate_properties(
    properties: Sequence[PropertyListing],
) -> List[PropertyListing]:
    """
    Remove duplicate listings. A listing is a duplicate if it has:
      - The same URL (after lowercasing), OR
      - The same (title + locality + city + rounded price) combination
        (catches the same property listed on multiple websites)
    """
    seen_urls: set = set()
    seen_keys: set = set()
    deduped: List[PropertyListing] = []

    for listing in properties:
        url_key = _normalize_text(listing.url)
        if url_key in seen_urls:
            continue
        seen_urls.add(url_key)

        key = (
            _normalize_text(listing.title),
            _normalize_text(listing.location.locality or ""),
            _normalize_text(listing.location.city or ""),
            round(listing.price or 0.0, -4) if listing.price else 0,
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(listing)

    return deduped


# SECTION 9 — SCORING
# Fast heuristic scores computed without calling an LLM.


def _investment_score(listing: PropertyListing) -> float:
    """
    Estimate investment value (0–100).
    Factors: price/sqft, city presence, key amenities.
    """
    score = 50.0

    if listing.price and listing.area_sqft:
        price_per_sqft = listing.price / max(listing.area_sqft, 1)
        if price_per_sqft < 10_000:
            score += 15
        elif price_per_sqft < 15_000:
            score += 8
        else:
            score -= 5

    if listing.location.city:
        score += 5

    amenities_lower = [a.lower() for a in listing.amenities]
    if any("metro" in a for a in amenities_lower):
        score += 5
    if any("parking" in a for a in amenities_lower):
        score += 4
    if any("gated" in a or "clubhouse" in a for a in amenities_lower):
        score += 4

    return float(max(min(score, 100.0), 0.0))


def _travel_score(listing: PropertyListing) -> float:
    """
    Estimate commute convenience (0–100).
    Factors: locality specificity, GPS availability, metro access.
    """
    score = 40.0

    if listing.location.locality:
        score += 15
    if listing.location.latitude is not None and listing.location.longitude is not None:
        score += 10
    if any("metro" in a.lower() for a in listing.amenities):
        score += 10

    return float(max(min(score, 100.0), 0.0))


def rank_properties(
    user_query: str,
    properties: Sequence[PropertyListing],
) -> List[PropertyListing]:
    """
    Assign 4 scores to every property using the NVIDIA NIM reranker, then sort best-first.
    Raises directly if the reranker call fails.
    """
    ranked: List[PropertyListing] = list(properties)
    if not ranked:
        return ranked

    docs = [
        Document(page_content=_property_text(p), metadata={"url": p.url})
        for p in ranked
    ]

    # Using the newer, actively supported Nemotron reranker
    url_to_score = {}
    try:
        reranker = NVIDIARerank(
            model="nvidia/llama-nemotron-rerank-1b-v2",
            nvidia_api_key=NVIDIA_API_KEY,
        )
        compressed = reranker.compress_documents(documents=docs, query=user_query)
        url_to_score = {
            doc.metadata["url"]: doc.metadata.get("relevance_score", 0.0)
            for doc in compressed
        }
    except Exception as e:
        print(
            f"⚠️ NVIDIA Reranker failed ({e}). Falling back to local semantic scoring."
        )

    for listing in ranked:
        listing.semantic_score = _score_semantic(user_query, _property_text(listing))
        listing.rerank_score = url_to_score.get(listing.url, 0.0)
        listing.investment_score = _investment_score(listing)
        listing.travel_score = _travel_score(listing)

    ranked.sort(
        key=lambda p: (
            p.rerank_score,
            p.semantic_score,
            p.investment_score,
            p.travel_score,
        ),
        reverse=True,
    )
    return ranked


# SECTION 10 — NEIGHBORHOOD ENRICHMENT (Google Maps)


def _google_places_nearby(
    lat: float,
    lng: float,
    keyword: str,
    radius_meters: int = 3000,
) -> List[NearbyPlace]:
    """
    Call Google Maps Nearby Search API to find places within a radius
    of the property's GPS coordinates. Returns up to 5 results.
    Raises RuntimeError if GOOGLE_MAPS_API_KEY is not set.
    Raises requests.HTTPError or requests.Timeout on network failure.
    """
    if not GOOGLE_MAPS_API_KEY:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set in .env")

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_meters,
        "keyword": keyword,
        "key": GOOGLE_MAPS_API_KEY,
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()

    return [
        NearbyPlace(name=item.get("name", "Unknown"), category=keyword, distance_km=0.0)
        for item in payload.get("results", [])[:5]
    ]


def enrich_neighborhood(
    listing: PropertyListing,
    preferences: Sequence[str],
) -> NeighborhoodAnalysis:
    """
    Fetch all categories of nearby places for one property.
    Gracefully skips the API call if the property has no GPS coordinates.
    """
    analysis = NeighborhoodAnalysis(property_url=listing.url)

    if listing.location.latitude is None or listing.location.longitude is None:
        return analysis

    lat, lng = listing.location.latitude, listing.location.longitude
    analysis.schools = _google_places_nearby(lat, lng, "school")
    analysis.hospitals = _google_places_nearby(lat, lng, "hospital")
    analysis.metro = _google_places_nearby(lat, lng, "metro station")
    analysis.malls = _google_places_nearby(lat, lng, "shopping mall")
    analysis.restaurants = _google_places_nearby(lat, lng, "restaurant")
    analysis.parks = _google_places_nearby(lat, lng, "park")
    return analysis


def _google_maps_text_search(query: str, limit: int = 5) -> NeighborhoodResult:
    """
    Free-text Google Maps Text Search API.
    Raises RuntimeError if GOOGLE_MAPS_API_KEY is not set.
    Raises requests.HTTPError or requests.Timeout on network failure.
    """
    if not GOOGLE_MAPS_API_KEY:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set in .env")

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": GOOGLE_MAPS_API_KEY}
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()

    places = [
        MapPlace(
            name=item.get("name", "Unknown"),
            category=query,
            address=item.get("formatted_address"),
        )
        for item in payload.get("results", [])[:limit]
    ]
    return NeighborhoodResult(query=query, places=places)


def neighborhood_lookup(query: str) -> NeighborhoodResult:
    """Public entry point for the neighborhood text search tool."""
    return _google_maps_text_search(query)


# SECTION 11 — PIPELINE ORCHESTRATION
# High-level functions that glue all the pieces together.

# The 6 biggest Indian real estate portals — searched on every query
TARGET_REAL_ESTATE_SITES = [
    "magicbricks.com",
    "99acres.com",
    "housing.com",
    "nobroker.in",
    "proptiger.com",
    "squareyards.com",
]


def scrape_properties_by_keyword(
    user_query: str, num_results: int = 10
) -> ScrapeBundle:
    """
    DATA LOADING — the entry point for fresh data.

    Flow:
      1. Generic web search (returns broad results)
      2. Site-specific searches on 6 Indian real estate portals
      3. Deduplicate all collected URLs
      4. Scrape each unique URL for full property details
      5. Normalize, deduplicate listings, store in MongoDB
    """
    # Step 1: Generic search
    raw_results: List[Dict] = search_service.search(user_query, num_results=num_results)

    # Step 2: Portal-specific searches — natural language works; site: operator is not supported
    # Each site name is appended to the query so ScrapeGraphAI targets that portal's results.
    for site in TARGET_REAL_ESTATE_SITES:
        site_name = site.split(".")[0]  # "magicbricks.com" → "magicbricks"
        site_query = f"{user_query} {site_name}"
        site_results = search_service.search(site_query, num_results=3)
        raw_results.extend(site_results)

    # Step 3: Deduplicate collected URLs before scraping
    seen_urls: set = set()
    unique_results: List[Dict] = []
    for item in raw_results:
        url = item.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(item)

    # Limit to top 8 URLs to avoid rate limits and extremely long wait times
    unique_results = unique_results[:8]
    print(
        f"Scraping top {len(unique_results)} unique URLs (capped at 8) from {len(TARGET_REAL_ESTATE_SITES) + 1} sources..."
    )

    # Step 4 & 5: Scrape, normalize, deduplicate, store
    listings: List[PropertyListing] = []
    for item in unique_results:
        url = item.get("url", "")
        if url:
            try:
                scraped = scraper.scrape(url)
                listings.append(normalize_listing(scraped))
            except Exception as e:
                print(f"⚠️ Skipping full scrape for {url} due to error: {e}")
                # Fall back to using the basic search snippet (title + url)
                listings.append(normalize_listing(item))
        else:
            listings.append(normalize_listing(item))

    listings = deduplicate_properties(listings)
    stored = mongo_store.upsert_many(listings)

    return ScrapeBundle(
        query=user_query,
        total_candidates=len(listings),
        stored_count=stored,
        urls=[listing.url for listing in listings],
    )


def retrieve_and_rerank(
    user_query: str,
    limit: int = 20,
    city: Optional[str] = None,
) -> List[PropertyListing]:
    """
    RETRIEVAL + RERANKING — fetch candidates from MongoDB (filtered by city if given),
    then rerank with NVIDIA NIM.
    Scrapes fresh data automatically if the database has no matches.
    """
    candidates = mongo_store.find_candidates(user_query, limit=limit, city=city)

    if not candidates:
        scrape_properties_by_keyword(user_query, num_results=limit)
        candidates = mongo_store.find_candidates(user_query, limit=limit, city=city)

    return rank_properties(user_query, candidates)


def search_properties(user_query: str, num_results: int = 10) -> SearchResult:
    """
    Full end-to-end search pipeline:
    parse query → log history → scrape → retrieve (city-filtered) → rerank → return
    """
    structured_query = parse_user_query(user_query)
    mongo_store.save_search_history(user_query)
    scrape_properties_by_keyword(user_query, num_results=num_results)
    # Pass the parsed city so retrieval only returns results from the right city
    properties = retrieve_and_rerank(
        user_query,
        limit=max(num_results, 20),
        city=structured_query.city,
    )
    properties = deduplicate_properties(properties)

    return SearchResult(
        query=structured_query,
        total_results=len(properties),
        properties=properties,
    )


def analyze_properties(
    user_query: str,
    properties: Sequence[PropertyListing],
) -> Tuple[List[PropertyListing], Dict[str, NeighborhoodAnalysis]]:
    """
    Rank all properties and enrich the top 10 with neighborhood data.
    Returns (ranked_list, {property_url: NeighborhoodAnalysis}).
    """
    ranked = rank_properties(user_query, properties)

    neighborhood_notes: Dict[str, NeighborhoodAnalysis] = {}
    for listing in ranked[:10]:
        analysis = enrich_neighborhood(listing, [])
        neighborhood_notes[listing.url] = analysis
        mongo_store.upsert_many([listing])  # persist the enriched listing

    return ranked, neighborhood_notes


def build_answer(
    user_query: str,
    ranked_properties: Sequence[PropertyListing],
    notes: Dict[str, NeighborhoodAnalysis],
) -> ChatResponse:
    """
    GENERATION — send top properties + neighborhood notes to Groq LLaMA
    and get a human-friendly recommendation summary back.
    """
    if not ranked_properties:
        return ChatResponse(
            answer=(
                "I could not retrieve matching listings from the configured sources. "
                "Try a broader query or add API keys for scraping and reranking."
            ),
            sources=[],
            recommended_properties=[],
        )

    top5_summary = [
        {
            "title": p.title,
            "url": p.url,
            "price": p.price,
            "city": p.location.city,
            "locality": p.location.locality,
            "beds": p.bedrooms,
            "score": p.rerank_score,
            "investment_score": p.investment_score,
        }
        for p in ranked_properties[:5]
    ]

    llm_text = _model_text(
        [
            {"role": "system", "content": ESTATEGPT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    ANSWER_PROMPT.format(query=user_query)
                    + "\n\nProperties:\n"
                    + json.dumps(top5_summary, indent=2, ensure_ascii=True)
                    + "\n\nNeighborhood notes:\n"
                    + json.dumps(
                        {k: v.model_dump(mode='json') for k, v in notes.items()},
                        indent=2,
                        ensure_ascii=True,
                    )
                ),
            },
        ],
        temperature=0.3,
    )

    if not llm_text:
        raise ValueError(f"LLM failed to generate a response for query: {user_query}")

    return ChatResponse(
        answer=llm_text,
        sources=[p.url for p in ranked_properties[:10]],
        recommended_properties=list(ranked_properties[:5]),
    )


def compare_properties(
    properties: Sequence[PropertyListing],
    user_query: str,
) -> ComparisonResult:
    """
    Compare up to 5 properties head-to-head.
    Reranks them first, then asks Groq LLaMA to pick the best with reasoning.
    """
    ranked = rank_properties(user_query, properties)
    best = ranked[0]

    summary = _model_text(
        [
            {"role": "system", "content": ESTATEGPT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    COMPARISON_PROMPT
                    + "\n\nProperties:\n"
                    + json.dumps(
                        [p.model_dump(mode='json') for p in ranked[:5]],
                        indent=2,
                        ensure_ascii=True,
                    )
                ),
            },
        ],
        temperature=0.25,
    )

    if not summary:
        summary = (
            f"{best.title} appears to be the strongest fit "
            "based on the available price, location, and amenity signals."
        )

    return ComparisonResult(
        best_property=best,
        compared_properties=list(ranked[:5]),
        summary=summary,
    )


def _direct_pipeline_response(user_query: str) -> ChatResponse:
    """
    Run the full pipeline WITHOUT the LangGraph agent.
    Used automatically when GOOGLE_API_KEY is not set.
    Auto-detects comparison intent from the query text.
    """
    result = search_properties(user_query, num_results=10)
    ranked, notes = analyze_properties(user_query, result.properties)

    comparison_keywords = ["compare", "comparison", "vs", "versus"]
    if any(word in user_query.lower() for word in comparison_keywords):
        comparison = compare_properties(ranked[:5], user_query)
        return ChatResponse(
            answer=comparison.summary,
            sources=[p.url for p in comparison.compared_properties],
            recommended_properties=list(comparison.compared_properties),
        )

    return build_answer(user_query, ranked, notes)


# SECTION 12 — EstateGPTAgent  (top-level class used by app.py)


class EstateGPTAgent:
    """
    The main agent class that app.py instantiates.

    Mode 1 — LangGraph Agent  (if GOOGLE_API_KEY is set)
        Gemini 1.5 Flash decides which tools to call and in what order.
        More flexible: handles complex multi-step reasoning.

    Mode 2 — Direct Pipeline  (if GOOGLE_API_KEY is NOT set)
        Fixed pipeline: scrape → rerank → answer.
        Always works without a Google key.
    """

    def __init__(self) -> None:
        self.tools = [estate_search, estate_compare, estate_neighborhood]
        self.agent = None

        if GOOGLE_API_KEY:
            self.agent = create_agent(
                model="google_genai:gemini-1.5-flash",
                tools=self.tools,
                system_prompt=ESTATEGPT_SYSTEM_PROMPT,
            )

    def invoke(self, user_query: str) -> ChatResponse:
        """
        Process a user query end-to-end and return a ChatResponse.
        Automatically picks Mode 1 or Mode 2 depending on config.
        """
        if self.agent is None:
            return _direct_pipeline_response(user_query)

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": user_query}]}
        )

        if isinstance(result, dict) and result.get("messages"):
            last_message = result["messages"][-1]
            output = getattr(last_message, "content", str(last_message))
        else:
            output = str(result)

        try:
            payload = _json_loads(output)
            return ChatResponse.model_validate(payload)
        except Exception:
            return ChatResponse(answer=output, sources=[], recommended_properties=[])


# CLI ENTRY POINT


def main() -> None:
    """
    CLI entry point for quick testing without the Streamlit UI.
    Usage: python estategpt.py 3BHK flat in Noida under 80 lakh near metro
    """
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        agent = EstateGPTAgent()
        response = agent.invoke(query)
        print(response.answer)
        return

    print("EstateGPT backend module loaded. Run app.py for the Streamlit UI.")


if __name__ == "__main__":
    main()
