# Where2Live Project

This project helps you find an Airbnb listing based on your preferences for price, school poverty rate, and nearby facilities. It uses a vector database (ChromaDB) to store and query listings, and a Large Language Model (LLM) to provide recommendations.

## Setup

1.  **Clone the repository:**
    Replace `<your-repository-url>` with the actual URL of this repository.
    ```bash
    git clone <your-repository-url>
    cd where2live
    ```

2.  **Create and activate a virtual environment:**
    It's recommended to use a virtual environment to manage project dependencies.

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

3.  **Install dependencies:**
    Make sure your virtual environment is activated, then run:
    ```bash
    pip3 install -r requirements.txt
    ```

4.  **Set up environment variables:**
    This project requires an API key for the GMI service.
    Create a file named `.env` in the root directory of the project (alongside `requirements.txt` and `app/`) with the following content:

    ```env
    GMI_API_KEY="YOUR_GMI_API_KEY_HERE"
    ```
    Replace `"YOUR_GMI_API_KEY_HERE"` with your actual GMI API key.

    The `.env` file is included in `.gitignore` and should not be committed to the repository.

## Optional: Testing your GMI API Key

If you want to test your `GMI_API_KEY` directly with the GMI service to ensure it's working correctly, you can use a `curl` command. This step is optional but can be helpful for troubleshooting.

1.  **Ensure your GMI_API_KEY is set** in the `.env` file in the project's root directory, as described in Step 4 of the "Setup" section.

2.  **Run the following command in your terminal** from the project's root directory. This command attempts to read the API key from your `.env` file:

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

    *   The `grep`, `cut`, and `sed` commands are used to extract the key value from the `.env` file.
    *   If you have `jq` installed (a command-line JSON processor), you can pipe the output to it for a more readable format: `... (previous curl command) | jq`
    *   A successful response will be a JSON object from the GMI service containing the LLM's reply (e.g., a list of countries and capitals). If there's an issue with your key or the request, you'll likely see an error message.

## Running the Application

1.  **Ensure your virtual environment is activated.**

2.  **Navigate to the application directory (if you are not already there):**
    The main application code is in the `app/` directory.

3.  **Run the FastAPI server using Uvicorn:**
    From the root directory of the project, run:
    ```bash
    uvicorn app.main:app --reload
    ```
    *   `app.main`: refers to the `main.py` file inside the `app` directory.
    *   `app`: refers to the `FastAPI` instance named `app` in `main.py`.
    *   `--reload`: enables auto-reloading, so the server will restart automatically when you make code changes.

4.  **Access the API:**
    Once the server is running, you can access the API documentation and interact with it at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your web browser.

    The main endpoint is `/suggest`. You can send a POST request to it with your preferences.

    Example using `curl`:
    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/suggest' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "max_price": 200,
      "max_school_pov": 30,
      "min_facilities": 2,
      "comments": "Looking for a quiet place near good schools"
    }'
    ```

## Project Structure

```
.
├── app/                  # Main application code
│   └── main.py           # FastAPI application
├── chroma_db/            # ChromaDB data (created automatically, in .gitignore)
├── .env                  # Environment variables (API keys, etc. - in .gitignore)
├── .gitignore            # Specifies intentionally untracked files that Git should ignore
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## Notes
* The first time you run the application, it might take a moment to download the sentence transformer model (`all-MiniLM-L6-v2`).
* The ChromaDB database will be created in the `chroma_db` directory if it doesn't already exist.