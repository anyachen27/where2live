from dotenv import load_dotenv
load_dotenv()
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

# New query model
class Query(BaseModel):
    weights: Dict[str, int]
    filters: Optional[Dict[str, Any]] = None
    comments: str = ""

app = FastAPI()

# Allow all origins for simpler local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GMI helper
GMI_API_KEY  = os.getenv("GMI_API_KEY")
GMI_ENDPOINT = "https://api.gmi-serving.com/v1/chat/completions"
GMI_MODEL    = "deepseek-ai/DeepSeek-Prover-V2-671B"

def gmi_chat(prompt: str) -> str:
    if not GMI_API_KEY:
        raise RuntimeError("GMI_API_KEY env var not set")
    payload = {
        "model": GMI_MODEL,
        "messages": [
            {"role": "system",
             "content": (
                "You are Where2Live, an expert housing advisor.\n\n"
                "TASK\n"
                "‣ Carefully analyze the *candidate_listings* JSON and the *user_prefs* JSON (including attribute weights, filters, and especially the user Comments/Preferences).\n"
                "‣ Use your world knowledge and reasoning to connect the user's Comments/Preferences to the house options. For example, if the user mentions being an engineer, prioritize houses with engineering-related industries. If they mention liking warm weather, use your knowledge of US zip codes/house addresses and climate to prefer warmer locations. Apply similar logic for other user preferences.\n"
                "‣ Select the 3 best house options from the candidates. For each, provide all fields from the dataset, the notes/description, and a clear, specific justification for why it was selected above the rest, referencing any relevant user preferences or inferred connections.\n"
                "‣ After the top 3, list the remaining candidate houses (raw hits) with their information and notes, but without justification.\n\n"
                "GENERAL PREFERENCES (unless user specifies otherwise):\n"
                "  • Lower price is always preferred.\n"
                "  • Lower poverty rate is always preferred.\n"
                "  • Higher school ranking is always preferred.\n"
                "  • Lower neighborhood crime rate is always preferred.\n"
                "  • Matching or similar common industries to the user's job or background is preferred.\n"
                "  • More beds and/or baths is only preferred if other statistics are similar, or if the user sets a hard filter for them. Otherwise, use beds/baths as a tiebreaker.\n\n"
                "RULES FOR EACH RECOMMENDATION OBJECT\n"
                "  • Include all listing fields and the notes/description.\n"
                "  • Add a justification (<=40 words) that is specific, references user preferences, and explains why this house is a better fit than the others.\n\n"
                "SELECTION CRITERIA\n"
                "  1. Must satisfy all hard limits in user_prefs.filters.\n"
                "  2. Among valid listings, rank by the weighted sum of attributes (higher weight = more important), but also use your reasoning to match user Comments/Preferences to house features and the general preferences above.\n"
                "  3. If <3 listings meet limits, return as many as possible.\n\n"
                "OUTPUT FORMAT (no deviation):\n"
                "{\n"
                "  \"top_3\": [\n"
                "    {\n"
                "      \"house_address\": \"...\",\n"
                "      \"zip_code\": \"...\",\n"
                "      \"price\": ...,\n"
                "      \"recreational_facilities\": \"...\",\n"
                "      \"poverty_rate\": ...,\n"
                "      \"number_of_beds\": ...,\n"
                "      \"number_of_baths\": ...,\n"
                "      \"school_ranking\": ...,\n"
                "      \"neighborhood_crime_rate\": ...,\n"
                "      \"common_industries\": \"...\",\n"
                "      \"other_notes\": \"...\",\n"
                "      \"justification\": \"...\"\n"
                "    }, ... (total 3) ...\n"
                "  ],\n"
                "  \"other_candidates\": [\n"
                "    {\n"
                "      \"house_address\": \"...\",\n"
                "      ... (all fields and notes, but no justification) ...\n"
                "    }, ...\n"
                "  ]\n"
                "}\n"
                "\n"
                "STRICT OUTPUT RULES:\n"
                "- Output ONLY the JSON object described above, with no extra text, comments, or explanations before or after.\n"
                "- All field values must use the correct type (numbers for numeric fields, strings for text).\n"
                "- Do NOT use trailing commas in arrays or objects.\n"
                "- Use plain text for all fields; do not use Unicode escape sequences unless absolutely necessary.\n"
                "- If no suitable listings are found, return \"top_3\": [] and \"other_candidates\": [].\n"
                "- The output must be valid JSON and must not contain any explanations, markdown, or formatting outside the JSON object."
                "- Double-check your output for valid JSON syntax before submitting. Do not include any explanations, comments, or markdown. Only output the JSON object."
            )},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1500
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

# Helper to build weighted query string
def build_weighted_query(user_weights: Dict[str, int], user_filters: Dict[str, Any], comments: str) -> str:
    # Build a string that emphasizes important attributes
    parts = []
    for k, w in user_weights.items():
        if w > 0:
            parts.append((f"{k} " * w).strip())
    filter_str = " ".join(f"{k}:{v}" for k, v in (user_filters or {}).items())
    return f"{comments} {filter_str} {' '.join(parts)}"

# Extract the first top-level JSON object from a string
def extract_json(text):
    import json

    start = text.find('{')
    if start == -1:
        return None
    # Use a stack to find matching closing brace
    stack = []
    for i in range(start, len(text)):
        if text[i] == '{':
            stack.append('{')
        elif text[i] == '}':
            stack.pop()
            if not stack:
                json_str = text[start:i+1]
                # Remove trailing commas before } or ]
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                return json_str
    return None

# Suggest endpoint
@app.post("/suggest")
def suggest(q: Query):
    try:
        # Weighted query string
        query_text = build_weighted_query(q.weights, q.filters, q.comments)
        qvec = embedder.encode(query_text).tolist()

        # Filter for ChromaDB
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
            # Only 1 filter, don't use $and
            k, v = next(iter(chroma_filter.items()))
            where = {k: v}
        elif len(chroma_filter) > 1:
            where = {"$and": [{k: v} for k, v in chroma_filter.items()]}
        else:
            where = None

        print("Received filters:", q.filters)
        print("Chroma filter:", chroma_filter)
        print("Sample ChromaDB metadata:", sample["metadatas"][0][0])  # from your startup print

        res = coll.query(
            query_embeddings=[qvec],
            n_results=25,
            where=where
        )
        metas_for_gmi = copy.deepcopy(res["metadatas"][0][:5])

        # If no raw hits, don't call LLM
        if not metas_for_gmi:
            return {
                "answer": [{"error": "No listings found matching your filtering criteria.", "details": "No raw hits available to pass to the LLM."}],
                "hits": []
            }

        # Mark lowest/highest
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

        # Prompt for LLM
        def build_prompt(user_params, rows):
            context = {row.get("House Address"): row for row in rows}
            return textwrap.dedent(f"""
                User preferences: {json.dumps(user_params)}.

                Below are candidate house listings (JSON). Rank the best 3 and justify in under 80 words. Mention all relevant attributes and user weights. Each object should include all fields and a justification.

                Listings:
                {json.dumps(context, indent=2)}
            """)

        prompt_for_gmi = build_prompt({"weights": q.weights, "filters": q.filters, "comments": q.comments}, metas_for_gmi)

        # LLM call + parsing output
        llm_recommendations = []
        try:
            raw_llm_output_str = gmi_chat(prompt_for_gmi)
            print("\n--- RAW LLM OUTPUT ---\n" + raw_llm_output_str + "\n--- END RAW LLM OUTPUT ---\n")
            cleaned_output = re.sub(r"^```(?:json)?\s*|```$", "", raw_llm_output_str.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
            print("CLEANED OUTPUT:\n", cleaned_output)
            json_str = extract_json(cleaned_output)
            print("EXTRACTED JSON STRING:\n", json_str)
            if not json_str:
                raise ValueError("No JSON object found in LLM output.")
            llm_output = json.loads(json_str)
            if not (isinstance(llm_output, dict) and "top_3" in llm_output and "other_candidates" in llm_output):
                raise ValueError("LLM output not in expected format (missing top_3/other_candidates).")
            llm_recommendations = llm_output
        except Exception as e:
            llm_recommendations = {"top_3": [], "other_candidates": [], "error": f"LLM did not return valid JSON or expected format: {e}"}

        return {"answer": llm_recommendations, "hits": metas_for_gmi}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
