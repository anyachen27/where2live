from dotenv import load_dotenv
load_dotenv()          # this reads .env into environment variables
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb, json, pathlib
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import traceback
import os, requests, textwrap
from typing import List, Dict, Any

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

# ---------- GMI helper --------------------------------------------------
GMI_API_KEY  = os.getenv("GMI_API_KEY")            # pulled from .env or shell
GMI_ENDPOINT = "https://api.gmi-serving.com/v1/chat/completions"
GMI_MODEL    = "deepseek-ai/DeepSeek-Prover-V2-671B"

def gmi_chat(prompt: str) -> str:
    if not GMI_API_KEY:
        raise RuntimeError("GMI_API_KEY env var not set")

    payload = {
        "model": GMI_MODEL,
        "messages": [
            {"role": "system",
             "content": "You are Where2Live, a housing advisor. "
                        "Always cite nightly price, school-poverty %, and facility count."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 400
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
        metas_for_gmi = res["metadatas"][0][:5] 

        prompt = build_prompt(q.dict(), metas_for_gmi)
        try:
            answer = gmi_chat(prompt)
        except Exception as e_gmi:
            # Graceful fallback if GMI API down
            answer = f"LLM summary unavailable. Error: {e_gmi}. Displaying top matches based on your filters:\n" + \
                     "\n".join(f"- {m['location']} (${m['price']}) poverty: {m['school_poverty']:.0f}% facilities: {m['facility_cnt']}" for m in metas_for_gmi)

        return {"answer": answer, "hits": metas_for_gmi} # Return the same hits GMI processed

    except Exception as e:
        traceback.print_exc()  # This will print to server logs
        raise HTTPException(status_code=500, detail=str(e))
