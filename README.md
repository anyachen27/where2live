# Where2Live Project

This project helps you find an Airbnb listing based on your preferences for price, school poverty rate, and nearby facilities. It uses a vector database (ChromaDB) to store and query listings, and a Large Language Model (LLM) to provide recommendations.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setup Instructions](#setup-instructions)
    - [Initial Environment Setup](#initial-environment-setup)
    - [Data Preparation and Database Population](#data-preparation-and-database-population)
3. [Running the Application](#running-the-application)
4. [Testing the Application](#testing-the-application)
5. [Optional: Testing your GMI API Key](#optional-testing-your-gmi-api-key)
6. [Project Structure](#project-structure)
7. [Important Notes](#important-notes)

## Prerequisites

*   Python 3.8+
*   Access to a GMI API key

## Setup Instructions

Follow these steps to set up and run the project on your local machine.

### Initial Environment Setup

1.  **Clone the Repository:**
    Replace `<your-repository-url>` with the actual URL of this repository.
    ```bash
    git clone <your-repository-url>
    cd where2live
    ```

2.  **Create and Activate a Virtual Environment:**
    Using a virtual environment is highly recommended.
    *   On macOS and Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *(Replace `venv` with `.venv` if you prefer)*

3.  **Install Dependencies:**
    With your virtual environment activated:
    ```bash
    pip3 install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    The application requires a GMI API key. Create a file named `.env` in the project's root directory with the following content:
    ```env
    GMI_API_KEY="YOUR_GMI_API_KEY_HERE"
    ```
    Replace `YOUR_GMI_API_KEY_HERE` with your actual key. The `.env` file is listed in `.gitignore` and should not be committed.

### Data Preparation and Database Population

This step involves preparing the listing data and populating the ChromaDB vector database. You only need to do this once, or when the source data changes.

1.  **Obtain the Data File:**
    You need the `sf_rag_housing.csv` file. This file contains the Airbnb listing data.
    *   **Action:** Place this file into the `data/processed/` directory in your project structure. Create these directories if they don't exist.

2.  **Generate JSON Blobs from CSV:**
    The `scripts/build_blobs.py` script processes the CSV file and creates individual JSON files (blobs) for each listing in the `blobs/` directory.
    Ensure your virtual environment is activated and you are in the project's root directory.
    ```bash
    python scripts/build_blobs.py
    ```
    This will create a `blobs/` directory (if it doesn't exist) filled with JSON files.

3.  **Embed Data and Populate ChromaDB:**
    The `scripts/embed_chroma.py` script takes the JSON blobs, generates embeddings for the relevant text data using a sentence transformer, and stores them along with metadata into the ChromaDB `where2live_airbnb` collection.
    Ensure your virtual environment is activated and you are in the project's root directory.
    ```bash
    python scripts/embed_chroma.py
    ```
    *   *(Note: The ChromaDB database will be stored locally in the `chroma_db/` directory, which is gitignored.)*

## Running the Application

Once the setup and data preparation are complete, you can run the FastAPI application server.

1.  **Ensure your virtual environment is activated.**
2.  **Start the FastAPI Server:**
    From the project's root directory:
    ```bash
    uvicorn app.main:app --reload
    ```
    *   `app.main`: Refers to the `main.py` file inside the `app/` directory.
    *   `app`: Refers to the `FastAPI` instance named `app` in `main.py`.
    *   `--reload`: Enables auto-reloading for development. The server restarts on code changes.

The server will typically be available at `http://127.0.0.1:8000`.

## Testing the Application

After starting the server, you can test the `/suggest` endpoint:

1.  **Using the Interactive API Docs (Swagger UI):**
    Open your web browser and navigate to `http://127.0.0.1:8000/docs`. You can interact with the API directly, try out the `/suggest` endpoint, and see responses.

2.  **Using `curl`:**
    Open a new terminal window and use a `curl` command. Example:
    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/suggest' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "max_price": 200,
      "max_school_pov": 900,
      "min_facilities": 2,
      "comments": "Looking for a quiet place near good schools"
    }'
    ```

3.  **Using the Test Script `test_api.py`:**
    The project includes a `test_api.py` script in the root directory. This script sends a predefined query to the `/suggest` endpoint.
    Ensure your virtual environment is activated and the FastAPI server is running. Then, in a new terminal, from the project root:
    ```bash
    python test_api.py
    ```
    This will print the API's response to the console.

## Optional: Testing your GMI API Key

If you want to test your `GMI_API_KEY` directly with the GMI service (before testing it through the application), use this `curl` command.

1.  **Ensure your `GMI_API_KEY` is set** in the `.env` file.
2.  **Run in your terminal** (from project root):
    ```bash
    GMI_API_KEY_FROM_ENV=$(grep GMI_API_KEY .env | cut -d '=' -f2 | sed 's/"//g')

    if [ -z "$GMI_API_KEY_FROM_ENV" ]; then
        echo "GMI_API_KEY not found in .env file or is empty. Please set it."
    else
        curl -sS https://api.gmi-serving.com/v1/chat/completions \
         -H "Content-Type: application/json" \
         -H "Authorization: Bearer $GMI_API_KEY_FROM_ENV" \
         -d '{
               "model":"deepseek-ai/DeepSeek-Prover-V2-671B",
               "messages":[
                 {"role":"system","content":"You are a helpful AI assistant."},
                 {"role":"user","content":"List 3 countries and their capitals."}
               ],
               "temperature":0.2,
               "max_tokens":300
             }'
    fi
    ```
    *   Use `... | jq` at the end if you have `jq` installed for formatted JSON output.
    *   A successful response is a JSON object from GMI.

## Project Structure

```
.
├── app/                  # Main FastAPI application code
│   └── main.py           # FastAPI app definition, endpoints
├── blobs/                # Stores JSON data blobs (generated by build_blobs.py, gitignored by default)
├── chroma_db/            # Local ChromaDB vector store (created automatically, gitignored)
├── data/
│   └── processed/
│       └── sf_rag_housing.csv # Source CSV data file (needs to be provided)
├── scripts/
│   ├── build_blobs.py    # Processes CSV into JSON blobs
│   └── embed_chroma.py   # Embeds blobs and populates ChromaDB
├── .env                  # Environment variables (API keys, etc. - gitignored)
├── .gitignore            # Specifies intentionally untracked files
├── README.md             # This file: setup and usage instructions
├── requirements.txt      # Python dependencies
└── test_api.py           # Basic script to test the /suggest API endpoint
```
*(Note: You might want to add `blobs/` to your `.gitignore` file if it's not already there and these files are intermediate.)*

## Important Notes

*   **First Run Model Download:** The first time `app/main.py` or `scripts/embed_chroma.py` runs (whichever uses `SentenceTransformer` first), it may take a moment to download the sentence transformer model (`all-MiniLM-L6-v2`).
*   **ChromaDB Location:** The ChromaDB vector database is stored locally in the `chroma_db/` directory. This directory will be created if it doesn't exist when the application or embedding script first interacts with ChromaDB.
*   **Data File `sf_rag_housing.csv`:** The setup assumes `data/processed/sf_rag_housing.csv` is available. Ensure you have this file and know its origin.