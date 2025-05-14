import pandas as pd
import numpy as np
import pathlib
import requests
import time
import json
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_CSV = PROCESSED_DIR / "sf_redfin_housing.csv"

# Redfin API endpoints
BASE_URL = "https://www.redfin.com"
SEARCH_URL = f"{BASE_URL}/stingray/do/location-autocomplete"
CSV_URL = f"{BASE_URL}/stingray/api/gis-csv"

# Browser-like headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0"
}

# San Francisco region ID (pre-determined)
SF_REGION = {
    "id": "20330",
    "type": "2",
    "name": "San Francisco"
}

def get_redfin_data(zip_code: int, page: int = 1) -> pd.DataFrame:
    """
    Fetch Redfin listings for a ZIP code.
    Uses San Francisco region ID and filters by ZIP code.
    """
    try:
        # Construct URL with SF region and ZIP filter
        params = {
            "al": "1",
            "market": "sf",
            "num_homes": "350",
            "ord": "redfin-recommended-asc",
            "page_number": str(page),
            "region_id": SF_REGION["id"],
            "region_type": SF_REGION["type"],
            "sf": "1,2,3,5,6,7",
            "status": "9",
            "uipt": "1,2,3,4,5,6,7,8",
            "v": "8",
            "zip": str(zip_code)
        }
        
        logging.info(f"Requesting data for ZIP {zip_code} page {page}")
        
        # Make request with full headers
        response = requests.get(
            CSV_URL,
            params=params,
            headers=HEADERS,
            timeout=30
        )
        
        if response.status_code != 200:
            logging.warning(f"Error fetching ZIP {zip_code}: HTTP {response.status_code}")
            return pd.DataFrame()
            
        # Save raw response for debugging
        raw_file = RAW_DIR / f"redfin_{zip_code}_p{page}.csv"
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        raw_file.write_text(response.text)
        
        # Check if we got actual CSV data
        if not response.text.strip().startswith('"'):  # CSVs typically start with a quote
            logging.warning(f"Response doesn't look like CSV data for ZIP {zip_code}")
            logging.debug(f"Response preview: {response.text[:200]}")
            return pd.DataFrame()
        
        # Parse CSV
        df = pd.read_csv(pd.compat.StringIO(response.text))
        if df.empty:
            logging.info(f"No listings found for ZIP {zip_code} page {page}")
            return df
            
        # Add ZIP code
        df['zip_code'] = zip_code
        logging.info(f"Found {len(df)} listings for ZIP {zip_code} page {page}")
        return df
        
    except Exception as e:
        logging.error(f"Error processing ZIP {zip_code}: {str(e)}")
        return pd.DataFrame()

def clean_redfin_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize Redfin data."""
    if df.empty:
        return df
        
    # Select and rename columns
    columns = {
        'MLS#': 'property_id',
        'LATITUDE': 'latitude',
        'LONGITUDE': 'longitude',
        'ZIP OR POSTAL CODE': 'zip_code',
        'PRICE': 'price',
        'BEDS': 'beds',
        'BATHS': 'baths',
        'LOCATION': 'location',
        'SQUARE FEET': 'sqft',
        'LOT SIZE': 'lot_size',
        'YEAR BUILT': 'year_built',
        'PROPERTY TYPE': 'property_type',
        'URL': 'url'
    }
    
    # Keep only columns we have
    available_cols = [col for col in columns.keys() if col in df.columns]
    df = df[available_cols].rename(columns={col: columns[col] for col in available_cols})
    
    # Clean numeric columns
    numeric_cols = ['price', 'beds', 'baths', 'sqft', 'year_built']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
    
    # Create description from available info
    desc_parts = []
    if 'property_type' in df.columns:
        desc_parts.append(df['property_type'].fillna('Property'))
    if 'beds' in df.columns:
        desc_parts.append(df['beds'].fillna(0).astype(int).astype(str) + ' bed')
    if 'baths' in df.columns:
        desc_parts.append(df['baths'].fillna(0).astype(int).astype(str) + ' bath')
    if 'sqft' in df.columns:
        desc_parts.append(df['sqft'].fillna(0).astype(int).astype(str) + ' sqft')
    if 'location' in df.columns:
        desc_parts.append('in ' + df['location'].fillna('San Francisco'))
    
    df['description'] = [' '.join(str(parts[i]) for i in range(len(parts))) 
                        for parts in zip(*desc_parts)]
    
    return df

def main():
    """Main script execution."""
    logging.info("Starting Redfin data collection...")
    
    # Start with just a few test ZIP codes
    test_zips = [94110, 94114, 94117]  # Mission, Castro, Haight-Ashbury
    
    all_listings = []
    for zip_code in test_zips:  # Use test_zips for initial testing
        logging.info(f"Processing ZIP {zip_code}")
        
        # Get first page
        page_df = get_redfin_data(zip_code, page=1)
        if not page_df.empty:
            all_listings.append(page_df)
            
            # If we got close to 350 results, try next page
            if len(page_df) > 300:
                logging.info(f"Fetching page 2 for ZIP {zip_code}")
                page2 = get_redfin_data(zip_code, page=2)
                if not page2.empty:
                    all_listings.append(page2)
        
        # Be nice to Redfin's servers
        time.sleep(3)
    
    # Combine all data
    if not all_listings:
        logging.error("No data collected!")
        return
        
    combined_df = pd.concat(all_listings, ignore_index=True)
    logging.info(f"Collected {len(combined_df)} total listings")
    
    # Clean data
    clean_df = clean_redfin_data(combined_df)
    
    # Save processed data
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUT_CSV, index=False)
    logging.info(f"Saved processed data to {OUT_CSV}")
    
    # Print summary
    print("\nData Summary:")
    print("-" * 40)
    print(f"Total listings: {len(clean_df)}")
    print("\nSample of collected data:")
    print(clean_df.head())
    print("\nColumns:")
    print(clean_df.columns.tolist())

if __name__ == "__main__":
    main() 