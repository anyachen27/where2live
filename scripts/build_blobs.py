import pandas as pd, json, pathlib, tqdm
import numpy as np

CSV_PATH = "data/processed/sf_rag_housing.csv"
BLOB_DIR = pathlib.Path("blobs")
BLOB_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="write blobs"):
    blob = {
        "id":            str(row.listing_id),
        "location":      row.location,
        "lat":           row.latitude,
        "lon":           row.longitude,
        "zip":           int(row.zip) if pd.notna(row.zip) else 0,  # handle NaN
        "district":      row.district_name,
        "price":         row.nightly_price,
        "school_poverty":row.school_poverty,
        "facility_cnt":  row.nearby_facilities,
        "notes":         str(row.notes)[:100]          # keep prompt small
    }
    (BLOB_DIR / f"{row.listing_id}.json").write_text(json.dumps(blob))

    