# Where2Live – Quick-Start Guide

Find Airbnb listings that balance **price**, **school poverty rate**, and **nearby facilities** using **ChromaDB** + an **LLM**!

## Prerequisites
- **Python 3.8+**
- `GMI_API_KEY` (add to `.env`)


## 1.  Clone & Set Up
```bash
git clone <your-repository-url>
cd where2live

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo 'GMI_API_KEY="YOUR_KEY"' > .env
```


## 2.  Prepare Data (one-time)
```bash
# build json blobs & populate chromadb
python scripts/build_blobs.py
python scripts/embed_chroma.py
```


## 3  Start Backend
```bash
source .venv/bin/activate
uvicorn app.main:app --reload       # http://127.0.0.1:8000/docs
```


## 4  Serve Frontend
_Open 2nd terminal, activate same venv, then:_
```bash
source .venv/bin/activate
python -m http.server 8080          # http://localhost:8080/index.html
```

<!-- ## 5  Test
- **Swagger UI:** open `/docs` in your browser.
- **cURL:**
  ```bash
  curl -X POST http://127.0.0.1:8000/suggest \
       -H "Content-Type: application/json" \
       -d '{"max_price":200,"max_school_pov":900,"min_facilities":2}'
  ``` -->

## (Optional) Reset Data
```bash
rm -rf blobs/* chroma_db/*
```


## Project Layout
```
app/            FastAPI API
scripts/        build_blobs.py, embed_chroma.py
data/processed/ sf_rag_housing.csv
blobs/          generated JSON (git-ignored)
chroma_db/      local vector DB (git-ignored)
index.html      simple frontend page
test_api.py     example request script
```

### Notes
- First run downloads the `all-MiniLM-L6-v2` model, allow some time
- `blobs/` and `chroma_db/` are git-ignored; delete them to rebuild with new data
