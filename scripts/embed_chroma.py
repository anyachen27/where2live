import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pathlib
import json
import re

CSV_PATH = "data/processed/us_house_listings.csv"
CHROMA_DIR = pathlib.Path("chroma_db")
COLLECTION_NAME = "where2live_houses"

def to_snake_case(s):
    return re.sub(r'[^a-zA-Z0-9]+', '_', str(s)).strip('_').lower()

# Read the new csv
df = pd.read_csv(CSV_PATH)
print("\nSample data from CSV:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# create or connect to local persistent chroma
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"\nDeleted existing collection '{COLLECTION_NAME}'")
except Exception:
    pass
coll = client.create_collection(COLLECTION_NAME)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Helper to create text representation of all fields for embedding
def weighted_text(row, weights=None):
    if weights is None:
        weights = {to_snake_case(col): 1 for col in row.index}
    text = []
    for k, v in row.items():
        k_snake = to_snake_case(k)
        w = weights.get(k_snake, 1)
        if w > 0 and isinstance(v, (str, int, float)):
            text.append(f"{k_snake}: {str(v)} " * int(w))
    return " ".join(text)

print(f"Embedding {len(df)} rows...")

for idx, row in df.iterrows():
    row_snake = {to_snake_case(k): v for k, v in row.items()}
    doc_id = f"{row_snake.get('house_address', idx)}_{idx}".replace("/", "_").replace(" ", "_")
    text = weighted_text(row)  # Now uses snake_case keys
    emb = embedder.encode(text).tolist()
    coll.add(
        documents=[text],
        embeddings=[emb],
        metadatas=[row_snake],
        ids=[doc_id]
    )

print("Done embedding rows.")
print("Chroma count:", coll.count())
