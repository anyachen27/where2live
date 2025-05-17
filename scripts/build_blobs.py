import pandas as pd, pathlib, json

CSV_PATH = "data/processed/us_house_listings.csv"
BLOB_DIR = pathlib.Path("blobs")
BLOB_DIR.mkdir(parents=True, exist_ok=True)

def to_snake_case(s):
    return s.strip().lower().replace(" ", "_").replace("-", "_")

df = pd.read_csv(CSV_PATH)
df.columns = [to_snake_case(col) for col in df.columns]

for _, row in df.iterrows():
    blob = {col: row[col] for col in df.columns}
    # Use address or a hash as unique id
    blob_id = str(row.get("house_address", str(_))).replace("/", "_").replace(" ", "_")
    (BLOB_DIR / f"{blob_id}.json").write_text(json.dumps(blob, ensure_ascii=False))
