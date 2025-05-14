# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb, json, pathlib
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import traceback

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

def llm_stub(rows, params):
    """Simple template until GMI chat is live."""
    lines = [f"Recommendations for params {params}:"]
    for r in rows:
        lines.append(
            f"- {r['location']} • ${r['price']}/night • "
            f"school poverty {r['school_poverty']:.1f}% • "
            f"{r['facility_cnt']} facilities nearby"
        )
    return "\n".join(lines)



@app.post("/suggest")
def suggest(q: Query):
    try:
        query_text = f"{q.comments} facilities {q.min_facilities}+ price {q.max_price}"
        qvec = embedder.encode(query_text).tolist()

        # First try without filters to see if we get any results
        print("\nTrying query without filters first...")
        no_filter_res = coll.query(
            query_embeddings=[qvec],
            n_results=5
        )
        print("Sample result without filters:", json.dumps(no_filter_res["metadatas"][0][0] if no_filter_res["metadatas"][0] else "No data", indent=2))

        # Then try with filters
        where = {
            "$and": [
                {"price": {"$lte": q.max_price}},
                {"school_poverty": {"$lte": q.max_school_pov}},
                {"facility_cnt": {"$gte": q.min_facilities}}
            ]
        }

        print(f"\nQuerying with conditions: {json.dumps(where, indent=2)}")
        res = coll.query(
            query_embeddings=[qvec],
            n_results=25,
            where=where
        )
        print(f"Query returned {len(res['metadatas'][0]) if res['metadatas'] else 0} results")
        
        if not res["metadatas"]:
            return {"answer": "No matches found for your criteria", "hits": []}
            
        metas = res["metadatas"][0]
        top5 = metas[:5]

        answer = llm_stub(top5, q.dict())
        return {"answer": answer, "hits": top5}
    except Exception as e:
        traceback.print_exc()  # This will print to server logs
        raise HTTPException(status_code=500, detail=str(e))
