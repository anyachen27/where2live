# scripts/embed_to_chroma.py
"""
Load the combined Airbnb–facilities CSV, embed each row, and upsert into Chroma
Columns expected in CSV:
listing_id, location, latitude, longitude, zip, district_name,
nightly_price, school_poverty, nearby_facilities, notes
"""

import pandas as pd, json, pathlib, tqdm, chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CSV = "data/processed/sf_rag_housing.csv"
BLOB_DIR = pathlib.Path("blobs")
CHROMA_DIR = pathlib.Path("chroma_db")
BLOB_DIR.mkdir(exist_ok=True, parents=True)
CHROMA_DIR.mkdir(exist_ok=True, parents=True)

# 1) read CSV
df = pd.read_csv(CSV)
print("\nSample data from CSV:")
print(df[["location", "nightly_price", "school_poverty", "nearby_facilities"]].head())
print("\nData types:")
print(df.dtypes)

# 2) create or connect to local persistent Chroma
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
# Delete existing collection if it exists
try:
    client.delete_collection("where2live_airbnb")
    print("\nDeleted existing collection")
except:
    pass
coll = client.create_collection("where2live_airbnb")

# 3) embedding model (MiniLM or swap in gmi_embed later)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

batch = 64
for i in tqdm.tqdm(range(0, len(df), batch), desc="embed rows"):
    chunk = df.iloc[i:i+batch]
    ids, docs, metas = [], [], []

    for _, r in chunk.iterrows():
        # Convert data types explicitly
        price = float(r.nightly_price)
        school_pov = float(r.school_poverty) if pd.notna(r.school_poverty) else 0.0
        facilities = int(r.nearby_facilities) if pd.notna(r.nearby_facilities) else 0
        
        ids.append(str(r.listing_id))
        meta = {
            "location": str(r.location),
            "price": price,
            "school_poverty": school_pov,
            "facility_cnt": facilities,
            "zip": str(r.zip)
        }
        metas.append(meta)
        
        # Print first few metadata entries for verification
        if len(metas) <= 5:
            print(f"\nSample metadata {len(metas)}:")
            print(json.dumps(meta, indent=2))
        
        docs.append(
            f"{r.location} {str(r.notes)[:150]} "
            f"price {price} facilities {facilities} "
            f"school_poverty {school_pov}"
        )

    vecs = embedder.encode(docs).tolist()
    coll.add(ids=ids, embeddings=vecs, metadatas=metas, documents=docs)

print("\nChroma count:", coll.count())

# Verify the data in Chroma
print("\nVerifying data in Chroma:")
sample = coll.query(
    query_embeddings=[[0] * 384],
    n_results=5
)
for i, meta in enumerate(sample["metadatas"][0]):
    print(f"\nSample {i+1} from Chroma:")
    print(json.dumps(meta, indent=2))
