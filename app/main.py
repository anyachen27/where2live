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
from typing import List, Dict, Any
import re
import copy

CHROMA_DIR = pathlib.Path("chroma_db")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
try:
    coll = client.get_collection("where2live_airbnb")
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
        coll = client.create_collection("where2live_airbnb")
        print("Created new collection")
    except Exception as e:
        print(f"Error creating collection: {e}")
        raise

embedder = SentenceTransformer("all-MiniLM-L6-v2")

class Query(BaseModel):
    max_price: float
    max_school_pov: float
    min_facilities: int = 0
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
            {
                "role": "system",
                "content": (
                    "You are Where2Live, an expert housing advisor.\n\n"
                    "TASK\n"
                    "‣ Examine the *candidate_listings* JSON and the *user_prefs* JSON.\n"
                    "‣ Return **exactly three** listings in a JSON array called `recommendations`—no extra keys, no prose outside the array.\n\n"
                    "RULES FOR EACH RECOMMENDATION OBJECT\n"
                    "  • `location`          – copy from listing\n"
                    "  • `price`             – nightly price (number)\n"
                    "  • `school_poverty`    – numeric value\n"
                    "  • `facility_cnt`      – numeric value\n"
                    "  • `notes`             – (optional) extra info about the listing, may be cited in justification if relevant\n"
                    "  • `justification`     – ≤30 words; cite price, school-poverty, facility_cnt, and notes if relevant. If a listing is the overall *lowest price*, *lowest school-poverty*, or *highest facility count*, say so (e.g. 'lowest price of all').\n\n"
                    "SELECTION CRITERIA\n"
                    "  1. Must satisfy all hard limits in *user_prefs* (max_price, max_school_pov, min_facilities).\n"
                    "  2. Among valid listings, rank by:\n"
                    "     • lowest price\n"
                    "     • THEN lowest school-poverty\n"
                    "     • THEN highest facility_cnt.\n"
                    "  3. If <3 listings meet limits, return as many as possible.\n\n"
                    "OUTPUT FORMAT (no deviation):\n"
                    "[\n"
                    "  {\n"
                    "    \"location\": \"…\",\n"
                    "    \"price\": 123,\n"
                    "    \"school_poverty\": 12.3,\n"
                    "    \"facility_cnt\": 4,\n"
                    "    \"notes\": \"…\",\n"
                    "    \"justification\": \"…\"\n"
                    "  }, … (total 3) …\n"
                    "]"
                )
            },
            { "role": "user", "content": prompt }
        ],
        "temperature": 0.0,  # Set to 0.0 for deterministic output
        "max_tokens": 600
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

# ---------- prompt builder ---------------------------------------------
def build_prompt(user_params: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    context = {row["location"]: {
        "price": row["price"],
        "school_poverty": row["school_poverty"],
        "facility_cnt": row["facility_cnt"]
    } for row in rows}

    return textwrap.dedent(f"""\
        User preferences: {json.dumps(user_params)}.

        Below are candidate Airbnb listings (JSON). Rank the best 3 and justify
        in under 80 words. Mention price, school-poverty %, and facility count.

        Listings:
        {json.dumps(context, indent=2)}
    """)


# def llm_stub(rows, params):
#     """Simple template until GMI chat is live."""
#     lines = [f"Recommendations for params {params}:"]
#     for r in rows:
#         lines.append(
#             f"- {r['location']} • ${r['price']}/night • "
#             f"school poverty {r['school_poverty']:.1f}% • "
#             f"{r['facility_cnt']} facilities nearby"
#         )
#     return "\\n".join(lines)



# @app.post("/suggest")
# def suggest(q: Query):
#     try:
#         qvec = embedder.encode(f"{q.comments} budget {q.max_price}").tolist()
#         filt = {
#             "price":          {"$lte": q.max_price},
#             "school_poverty": {"$lte": q.max_school_pov},
#             "facility_cnt":   {"$gte": q.min_facilities}
#         }
#         res   = coll.query(query_embeddings=[qvec], n_results=30, filter=filt)
#         metas = res["metadatas"][0][:5]          # top-5

#         prompt = build_prompt(q.dict(), metas)
#         try:
#             answer = gmi_chat(prompt)
#         except Exception as e:
#             # graceful fallback if API down
#             answer = "LLM unavailable.\\n" + \\
#                      "\\n".join(f"- {m['location']} (${m['price']})" for m in metas)

#         return {"answer": answer, "hits": metas}

#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))
    

# The (previously commented out) suggest function that uses 'where'
# will be uncommented and corrected in the next step.

@app.post("/suggest")
def suggest(q: Query):
    try:
        query_text = f"{q.comments} facilities {q.min_facilities}+ price {q.max_price}"
        qvec = embedder.encode(query_text).tolist()

        # Optional: query without filters first to see if GMI should be called at all
        # print("\nTrying query without filters first...")
        # no_filter_res = coll.query(
        #     query_embeddings=[qvec],
        #     n_results=5
        # )
        # print("Sample result without filters:", json.dumps(no_filter_res["metadatas"][0][0] if no_filter_res["metadatas"] and no_filter_res["metadatas"][0] else "No data", indent=2))

        where_conditions = {
            "$and": [
                {"price": {"$lte": q.max_price}},
                {"school_poverty": {"$lte": q.max_school_pov}},
                {"facility_cnt": {"$gte": q.min_facilities}}
            ]
        }

        # print(f"\nQuerying with conditions: {json.dumps(where_conditions, indent=2)}")
        res = coll.query(
            query_embeddings=[qvec],
            n_results=25, # Fetch more for GMI to pick from, GMI will select top 3
            where=where_conditions
        )
        # print(f"Query returned {len(res['metadatas'][0]) if res['metadatas'] and res['metadatas'][0] else 0} results")
        
        if not res["metadatas"] or not res["metadatas"][0]:
            answer = "No listings found matching your filtering criteria."
            return {"answer": answer, "hits": []}
            
        # Pass top 5 relevant (and filtered) listings to GMI
        metas_for_gmi = copy.deepcopy(res["metadatas"][0][:5])
        if metas_for_gmi:
            min_price = min(m['price'] for m in metas_for_gmi)
            min_school_poverty = min(m['school_poverty'] for m in metas_for_gmi)
            max_facility_cnt = max(m['facility_cnt'] for m in metas_for_gmi)
            for m in metas_for_gmi:
                notes = []
                if m['price'] == min_price:
                    notes.append("lowest price")
                if m['school_poverty'] == min_school_poverty:
                    notes.append("lowest school poverty")
                if m['facility_cnt'] == max_facility_cnt:
                    notes.append("highest facility count")
                m['note'] = ", ".join(notes) if notes else ""
                # Ensure 'notes' field is present for LLM context
                if 'notes' not in m:
                    m['notes'] = ""

        prompt_for_gmi = build_prompt(q.dict(), metas_for_gmi)
        llm_recommendations = [] # To store the list of recommendation dicts

        try:
            raw_llm_output_str = gmi_chat(prompt_for_gmi)
            # Remove Markdown code block formatting if present (triple backticks, optional 'json' label, case-insensitive)
            cleaned_output = re.sub(r"^```(?:json)?\s*|```$", "", raw_llm_output_str.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
            try:
                # Attempt to parse the cleaned LLM's string output as JSON
                llm_recommendations = json.loads(cleaned_output)
                if not isinstance(llm_recommendations, list): # Basic validation
                    print(f"LLM output was valid JSON but not a list: {llm_recommendations}")
                    raise ValueError("LLM output not a list as expected.")
            except json.JSONDecodeError as e_json:
                print(f"LLM output was not valid JSON. Error: {e_json}. Raw output: {raw_llm_output_str}")
                # Fallback: use the raw string if it's not JSON, or create a custom error structure
                llm_recommendations = [{"error": "LLM did not return valid JSON.", "details": raw_llm_output_str}]
            except ValueError as e_val: # Handles the "not a list" case
                 llm_recommendations = [{"error": "LLM output not in expected format.", "details": str(e_val)}]

        except Exception as e_gmi:
            # Graceful fallback if GMI API down or other GMI call error
            print(f"GMI chat error: {e_gmi}")
            llm_recommendations = [{
                "error": "LLM summary unavailable.",
                "details": str(e_gmi),
                "fallback_message": "Displaying top matches based on your filters.",
                "matches": [
                    {
                        "location": m['location'],
                        "price": m['price'],
                        "school_poverty": f"{m['school_poverty']:.0f}%",
                        "facility_cnt": m['facility_cnt'],
                        "justification": "Direct match based on filters (LLM unavailable)."
                    } for m in metas_for_gmi
                ]
            }]

        return {"answer": llm_recommendations, "hits": metas_for_gmi} # Return the structured recommendations

    except Exception as e:
        traceback.print_exc()  # This will print to server logs
        raise HTTPException(status_code=500, detail=str(e))
