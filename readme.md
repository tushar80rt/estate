<div align="center">

# 🏠 EstateGPT

**Intelligent Real Estate Matchmaking Engine for India**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59.1-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1.3.13-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.17.0-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-NIM_APIs-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://build.nvidia.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/tushar80rt/estate?style=for-the-badge&color=gold)](https://github.com/tushar80rt/estate/stargazers)

> An agentic AI platform that **scrapes**, **embeds**, **reranks**, and **explains** real estate listings using a full RAG pipeline — all from a single natural-language query.

</div>

---

## ✨ What is EstateGPT?

EstateGPT is a **production-grade AI real-estate assistant** that turns a plain-English property request like:

> *"3 BHK flat in Bangalore under ₹1.2 Cr with metro access and a good school nearby"*

...into a **ranked, enriched, explainable shortlist** of matching properties — scraped live from India's top real-estate portals.

---

## 🎬 Features at a Glance

| Feature | Description |
|---|---|
| 🔍 **Natural Language Search** | Describe what you want in plain English |
| 🏆 **AI-Powered Ranking** | Properties ranked by NVIDIA NIM reranker |
| 🗺️ **Neighborhood Intelligence** | Nearby schools, hospitals, metro via Google Maps |
| ⚖️ **Property Comparison** | Side-by-side AI comparison with verdict |
| 💾 **Search History** | MongoDB-persisted history in the sidebar |
| 📊 **Investment Score** | Heuristic investment & travel suitability scores |
| 🔗 **Multi-Source Scraping** | Scrapes MagicBricks, 99acres & more |

---

## 🏗️ Architecture

### Full RAG Pipeline

```mermaid
flowchart TD
    A([👤 User Query]) --> B["🔤 Query Parser
    Groq LLaMA-3.3-70B"]
    B --> C{Structured
    SearchQuery}

    C --> D["🌐 Web Scraper
    ScrapeGraphAI SmartScraper"]
    D --> E[📄 Raw Listings]

    E --> F["🧹 Normalize and Deduplicate
    backend.py"]
    F --> G[("🗄️ MongoDB
    Property Store")]

    G --> H["✂️ Text Chunker
    RecursiveCharacterTextSplitter"]
    H --> I["🧲 NVIDIA Embeddings
    nv-embedqa-e5-v5"]
    I --> J[("📦 MongoDB
    Chunk Store")]

    J --> K["🔎 Cosine Similarity Retrieval"]
    K --> L["🏅 NVIDIA NIM Reranker
    nv-rerankqa-mistral-4b-v3"]
    L --> M["🗺️ Neighborhood Lookup
    Google Maps API"]

    M --> N["🤖 Answer Generator
    Groq LLaMA-3.3-70B"]
    N --> O([💬 Ranked Results + Explanation])
```

---

### Project Module Map

```mermaid
graph LR
    subgraph UI["🖥️ Frontend"]
        A["app.py · Streamlit UI"]
    end

    subgraph Core["⚙️ Core Engine"]
        B["backend.py · RAG Pipeline"]
        C["scrapper.py · Web Scraper"]
        D["models.py · Pydantic Models"]
        E["config.py · Env Settings"]
    end

    subgraph Storage["🗄️ MongoDB Collections"]
        F[("properties")]
        G[("property_chunks")]
        H[("search_history")]
    end

    subgraph APIs["🔌 External APIs"]
        I["Groq · LLaMA-3.3-70B"]
        J["NVIDIA NIM · Embeddings + Reranker"]
        K["ScrapeGraphAI · SmartScraper"]
        L["Google Maps · Nearby Places"]
    end

    A --> B
    B --> C
    B --> D
    B --> E
    B --> F
    B --> G
    B --> H
    B --> I
    B --> J
    C --> K
    B --> L
```

---

### Agent Tool Sequence

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant A as 🖥️ Streamlit App
    participant AG as 🤖 LangGraph Agent
    participant T1 as 🔍 estate_search
    participant T2 as ⚖️ estate_compare
    participant T3 as 🗺️ estate_neighborhood

    U->>A: "Find me a 2BHK in Pune under 80L"
    A->>AG: invoke(query)
    AG->>T1: estate_search(query)
    T1-->>AG: Ranked Properties JSON
    AG->>T3: estate_neighborhood(query)
    T3-->>AG: Nearby Places JSON
    AG-->>A: ChatResponse
    A-->>U: Property cards + AI explanation

    Note over U,A: Compare mode
    U->>A: "Which is better — A or B?"
    A->>AG: invoke(query)
    AG->>T2: estate_compare(query)
    T2-->>AG: Comparison Result JSON
    AG-->>A: ChatResponse
    A-->>U: Side-by-side verdict
```

---

## 🚀 Quick Start

### Prerequisites

| Tool | Minimum Version | Download |
|---|---|---|
| Python | 3.10+ | [python.org](https://python.org) |
| MongoDB | 6.0+ | [mongodb.com](https://mongodb.com/try/download/community) |
| Git | any | [git-scm.com](https://git-scm.com) |

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/tushar80rt/estate.git
cd estate
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure API Keys

Create a `.env` file in the project root:

```env
# ── LLM  (Required) ────────────────────────────────────────────────
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.3-70b-versatile

# ── Web Scraping  (Required) ────────────────────────────────────────
SCRAPEGRAPH_API_KEY=sgai_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ── NVIDIA NIM — Embeddings & Reranker  (Required) ─────────────────
NVIDIA_API_KEY=nvapi_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ── MongoDB  (Required) ─────────────────────────────────────────────
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=estategpt
MONGODB_COLLECTION=properties
MONGODB_CHUNK_COLLECTION=property_chunks

# ── Google Maps  (Optional — for neighborhood intelligence) ─────────
GOOGLE_MAPS_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **💡 Tip:** You can skip the `.env` file entirely and paste your **Groq** and **ScrapeGraph** keys directly in the Streamlit sidebar at runtime.

### Step 5 — Start MongoDB

```bash
# Windows (service)
net start MongoDB

# macOS / Linux
mongod --dbpath /data/db
```

### Step 6 — Launch the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser 🎉

---

## 🔑 API Keys Reference

| Service | Used For | Free Tier | Get It |
|---|---|---|---|
| **Groq** | LLM inference (LLaMA-3.3-70B) | ✅ Yes | [console.groq.com](https://console.groq.com) |
| **ScrapeGraphAI** | AI-powered web scraping | ✅ Yes | [scrapegraphai.com](https://scrapegraphai.com) |
| **NVIDIA NIM** | Embeddings + Neural reranking | ✅ Free credits | [build.nvidia.com](https://build.nvidia.com) |
| **Google Maps** | Neighborhood places & distances | ✅ $200 credit/mo | [console.cloud.google.com](https://console.cloud.google.com) |

---

## 📁 Project Structure

```
estate/
│
├── app.py              # 🖥️  Streamlit UI — chat input, property cards, sidebar history
├── backend.py          # ⚙️  Core RAG engine — scrape → embed → rerank → answer
├── scrapper.py         # 🌐  ScrapeGraphAI SmartScraper integration
├── models.py           # 📐  All Pydantic v2 data models
├── config.py           # 🔧  Environment variables & constants
│
├── assets/
│   └── Groq.svg        # 🎨  Groq logo used in sidebar
│
├── requirements.txt    # 📦  Pinned Python dependencies
└── .env                # 🔒  Secret keys — NOT committed to Git
```

---

## 🧠 How It Works — 8-Step Pipeline

```mermaid
flowchart LR
    Q[/"📝 User types a
    property request"/]

    Q --> S1["1️⃣ PARSE
    Groq extracts city,
    beds, budget, amenities"]

    S1 --> S2["2️⃣ SCRAPE
    ScrapeGraphAI hits
    MagicBricks, 99acres"]

    S2 --> S3["3️⃣ STORE
    Normalized listings
    saved to MongoDB"]

    S3 --> S4["4️⃣ EMBED
    NVIDIA nv-embedqa-e5-v5
    vectorizes each chunk"]

    S4 --> S5["5️⃣ RETRIEVE
    Cosine similarity search
    over chunk embeddings"]

    S5 --> S6["6️⃣ RERANK
    NVIDIA NIM reranker
    scores top candidates"]

    S6 --> S7["7️⃣ ENRICH
    Google Maps finds
    nearby schools & metro"]

    S7 --> S8["8️⃣ ANSWER
    Groq LLaMA writes
    pros, cons & verdict"]

    S8 --> R[/"✅ Ranked property cards
    displayed to user"/]
```

---

## ⚙️ Configuration Reference

| Variable | Default | Required | Description |
|---|---|:---:|---|
| `GROQ_API_KEY` | — | ✅ | Groq API key for LLM inference |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | ❌ | Groq model identifier |
| `SCRAPEGRAPH_API_KEY` | — | ✅ | ScrapeGraphAI SmartScraper key |
| `NVIDIA_API_KEY` | — | ✅ | NVIDIA NIM key (embeddings + reranker) |
| `GOOGLE_MAPS_API_KEY` | — | ❌ | Google Maps Places API key |
| `MONGODB_URI` | `mongodb://localhost:27017` | ✅ | MongoDB connection string |
| `MONGODB_DB` | `estategpt` | ❌ | Target database name |
| `MONGODB_COLLECTION` | `properties` | ❌ | Property documents collection |
| `MONGODB_CHUNK_COLLECTION` | `property_chunks` | ❌ | RAG chunk embeddings collection |

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **UI** | Streamlit | 1.59.1 |
| **LLM** | Groq — LLaMA-3.3-70B-Versatile | 1.5.0 |
| **Embeddings** | NVIDIA `nv-embedqa-e5-v5` | NIM |
| **Reranker** | NVIDIA `nv-rerankqa-mistral-4b-v3` | NIM |
| **Web Scraping** | ScrapeGraphAI SmartScraper | 2.1.0 |
| **Agent Framework** | LangChain + LangGraph | 1.3.13 / 1.2.9 |
| **Vector Search** | MongoDB cosine similarity | 4.17.0 |
| **Data Models** | Pydantic v2 | 2.13.4 |
| **Neighborhood** | Google Maps Places API | v1 |

---

## 🤝 Contributing

Contributions are welcome!

1. **Fork** this repo
2. Create a feature branch — `git checkout -b feature/your-feature`
3. Commit your changes — `git commit -m "Add your feature"`
4. Push to your branch — `git push origin feature/your-feature`
5. Open a **Pull Request** and describe what you changed

Please follow [PEP 8](https://peps.python.org/pep-0008/) and add docstrings to any new functions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Tushar](https://github.com/tushar80rt)**

If this project helped you, please consider giving it a ⭐ — it means a lot!

[![GitHub Stars](https://img.shields.io/github/stars/tushar80rt/estate?style=social)](https://github.com/tushar80rt/estate)

</div>
