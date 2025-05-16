from dotenv import load_dotenv
load_dotenv()          # this reads .env into environment variables
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb, json, pathlib
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import traceback
import os, requests, textwrap
from typing import List, Dict, Any, Optional
import re
import copy

CHROMA_DIR = pathlib.Path("chroma_db")
COLLECTION_NAME = "where2live_houses"
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
try:
    coll = client.get_collection(COLLECTION_NAME)
    print(f"Successfully connected to collection. Count: {coll.count()}")
    # Get a sample of the data to verify its structure
    sample = coll.query(
        query_embeddings=[[0] * 384],  # MiniLM produces 384-dimensional embeddings
        n_results=5,  # Get more samples
        where=None  # No filtering
    )
    print("\nSample metadata:")
    for meta in sample["metadatas"][0]:
        print(json.dumps(meta, indent=2))
except Exception as e:
    print(f"Error getting collection: {e}")
    # Try to create it if it doesn't exist
    try:
        coll = client.create_collection(COLLECTION_NAME)
        print("Created new collection")
    except Exception as e:
        print(f"Error creating collection: {e}")
        raise

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- New Query Model ---
class Query(BaseModel):
    weights: Dict[str, int]
    filters: Optional[Dict[str, Any]] = None
    comments: str = ""

app = FastAPI()

# CORS Middleware Configuration
# Allow all origins for simplicity in local development.
# For production, you should restrict this to your frontend's actual origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# ---------- GMI helper --------------------------------------------------
GMI_API_KEY  = os.getenv("GMI_API_KEY")            # pulled from .env or shell
GMI_ENDPOINT = "https://api.gmi-serving.com/v1/chat/completions"
GMI_MODEL    = "deepseek-ai/DeepSeek-Prover-V2-671B"

def gmi_chat(prompt: str) -> str:  # Return type is still str, as LLM outputs JSON string
    if not GMI_API_KEY:
        raise RuntimeError("GMI_API_KEY env var not set")

    payload = {
        "model": GMI_MODEL,
        "messages": [
            {"role": "system",
             "content": (
                "You are Where2Live, an expert housing advisor.\n\n"
                "TASK\n"
                "‣ Examine the *candidate_listings* JSON and the *user_prefs* JSON (including attribute weights and filters).\n"
                "‣ Return exactly three listings in a JSON array called recommendations—no extra keys, no prose outside the array.\n\n"
                "RULES FOR EACH RECOMMENDATION OBJECT\n"
                "  • Include all listing fields.\n"
                "  • Add a justification (≤30 words) referencing the most important attributes (by user weight) and any filters applied.\n\n"
                "SELECTION CRITERIA\n"
                "  1. Must satisfy all hard limits in user_prefs.filters.\n"
                "  2. Among valid listings, rank by the weighted sum of attributes (higher weight = more important).\n"
                "  3. If <3 listings meet limits, return as many as possible.\n\n"
                "OUTPUT FORMAT (no deviation):\n"
                "[\n"
                "  {\n"
                "    \"house_address\": \"...\",\n"
                "    \"zip_code\": \"...\",\n"
                "    \"price\": ...,\n"
                "    \"recreational_facilities\": \"...\",\n"
                "    \"poverty_rate\": ...,\n"
                "    \"number_of_beds\": ...,\n"
                "    \"number_of_baths\": ...,\n"
                "    \"school_ranking\": ...,\n"
                "    \"neighborhood_crime_rate\": ...,\n"
                "    \"common_industries\": \"...\",\n"
                "    \"other_notes\": \"...\",\n"
                "    \"justification\": \"...\"\n"
                "  }, ... (total 3) ...\n"
                "]"
            )},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,  # Set to 0.0 for deterministic output
        "max_tokens": 800
    }

    r = requests.post(
        GMI_ENDPOINT,
        headers={"Authorization": f"Bearer {GMI_API_KEY}",
                 "Content-Type": "application/json"},
        json=payload, timeout=60
    )
    if r.status_code != 200:
        raise RuntimeError(f"GMI error {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"].strip()

# --- Helper to build weighted query string ---
def build_weighted_query(user_weights: Dict[str, int], user_filters: Dict[str, Any], comments: str) -> str:
    # Build a string that emphasizes important attributes
    parts = []
    for k, w in user_weights.items():
        if w > 0:
            parts.append((f"{k} " * w).strip())
    filter_str = " ".join(f"{k}:{v}" for k, v in (user_filters or {}).items())
    return f"{comments} {filter_str} {' '.join(parts)}"

# --- Suggest endpoint ---
@app.post("/suggest")
def suggest(q: Query):
    try:
        # Build weighted query string
        query_text = build_weighted_query(q.weights, q.filters, q.comments)
        qvec = embedder.encode(query_text).tolist()

        # Build filter for ChromaDB
        chroma_filter = {}
        if q.filters:
            for k, v in q.filters.items():
                if isinstance(v, (int, float)):
                    # Guess filter type by key
                    if k.startswith("max_"):
                        field = k[4:]
                        chroma_filter[field] = {"$lte": v}
                    elif k.startswith("min_"):
                        field = k[4:]
                        chroma_filter[field] = {"$gte": v}
                    else:
                        chroma_filter[k] = v
                else:
                    chroma_filter[k] = v

        if len(chroma_filter) == 1:
            # Only one filter, don't use $and
            k, v = next(iter(chroma_filter.items()))
            where = {k: v}
        elif len(chroma_filter) > 1:
            where = {"$and": [{k: v} for k, v in chroma_filter.items()]}
        else:
            where = None

        res = coll.query(
            query_embeddings=[qvec],
            n_results=25,
            where=where
        )
        metas_for_gmi = copy.deepcopy(res["metadatas"][0][:5])

        # Annotate for lowest/highest
        if metas_for_gmi:
            # Only consider numeric fields for lowest/highest
            numeric_fields = [k for k, v in metas_for_gmi[0].items() if isinstance(v, (int, float))]
            for field in numeric_fields:
                values = [m[field] for m in metas_for_gmi]
                min_val = min(values)
                max_val = max(values)
                for m in metas_for_gmi:
                    notes = []
                    if m[field] == min_val:
                        notes.append(f"lowest {field}")
                    if m[field] == max_val:
                        notes.append(f"highest {field}")
                    m.setdefault("note", "")
                    if notes:
                        m["note"] += (", " if m["note"] else "") + ", ".join(notes)
            for m in metas_for_gmi:
                if "notes" not in m:
                    m["notes"] = ""

        # Build prompt for LLM
        def build_prompt(user_params, rows):
            context = {row["house_address"]: row for row in rows}
            return textwrap.dedent(f"""
                User preferences: {json.dumps(user_params)}.

                Below are candidate house listings (JSON). Rank the best 3 and justify in under 80 words. Mention all relevant attributes and user weights. Each object should include all fields and a justification.

                Listings:
                {json.dumps(context, indent=2)}
            """)

        prompt_for_gmi = build_prompt({"weights": q.weights, "filters": q.filters, "comments": q.comments}, metas_for_gmi)

        # --- LLM call and output parsing ---
        llm_recommendations = []
        try:
            raw_llm_output_str = gmi_chat(prompt_for_gmi)
            cleaned_output = re.sub(r"^```(?:json)?\s*|```$", "", raw_llm_output_str.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
            llm_recommendations = json.loads(cleaned_output)
            if not isinstance(llm_recommendations, list):
                raise ValueError("LLM output not a list as expected.")
        except Exception as e:
            llm_recommendations = [{"error": "LLM did not return valid JSON.", "details": str(e)}]

        return {"answer": llm_recommendations, "hits": metas_for_gmi}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
