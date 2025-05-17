# Purpose: create a CSV for RAG for Where2Live from raw datasets
# 1. listings.csv (InsideAirbnb - San Francisco) - was going to use house listings but they were very hard to find
# 2. San_Francisco_ZIP_Codes_20250514.csv  (zip data)
# 3. School_District_Boundaries_-_Current.csv  (district data)
# 4. School_Neighborhood_Poverty_Estimates_Current_-8025457622334316475.csv (point schools)
# 5. Recreation_and_Parks_Properties_20250514.csv (parks/recreation facilities)

import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
from shapely.geometry import Point, Polygon
import numpy as np
import pathlib, json, re

DATA_DIR = pathlib.Path("data")
RAW_DIR  = DATA_DIR / "raw"
OUT_CSV  = DATA_DIR / "processed" / "sf_rag_housing.csv"

LISTINGS_CSV   = RAW_DIR / "listings.csv"
ZIP_DATA       = RAW_DIR / "San_Francisco_ZIP_Codes_20250514.csv"
DIST_DATA      = RAW_DIR / "School_District_Boundaries_-_Current.csv"
POVERTY_CSV    = RAW_DIR / "School_Neighborhood_Poverty_Estimates_Current_-8025457622334316475.csv"
PARKS_CSV      = RAW_DIR / "Recreation_and_Parks_Properties_20250514.csv"

# parameters
POVERTY_RADIUS_KM = 2          # radius to average nearby school‑poverty %
PARKS_RADIUS_KM = 1            # radius to count nearby parks/facilities

# Load Airbnb listings (only keep essential stuff)
print("Loading Airbnb listings …")
bnb = pd.read_csv(LISTINGS_CSV, low_memory=False,
                  usecols=["id","name","neighbourhood",
                           "latitude","longitude","price"])

# Clean nightly price
bnb["price"] = (
    bnb["price"]
      .astype(str)
      .str.replace(r"[^\d.]", "", regex=True)
      .replace("", np.nan)
      .astype(float)
)

bnb = bnb.dropna(subset=["price"])

# Create geodataframe
bnb_gdf = gpd.GeoDataFrame(
    bnb,
    geometry=gpd.points_from_xy(bnb.longitude, bnb.latitude),
    crs="EPSG:4326"
)

# Load zip data
print("Loading ZIP data …")
try:
    zip_df = pd.read_csv(ZIP_DATA)
    print("ZIP data columns:", zip_df.columns.tolist())
    
    # Identify the zip code column
    zip_columns = [col for col in zip_df.columns if 'zip' in col.lower()]
    if not zip_columns:
        raise ValueError("Could not find ZIP code column in data")
    zip_column = zip_columns[0]
    
    # Clean up zip data
    zip_df = zip_df.rename(columns={zip_column: 'zip'})
    if 'geometry' in zip_df.columns:
        zip_gdf = gpd.GeoDataFrame(zip_df, geometry=gpd.GeoSeries.from_wkt(zip_df['geometry']))
        zip_gdf.crs = "EPSG:4326"
    else:
        zip_gdf = zip_df
    
except Exception as e:
    print(f"Error loading ZIP data: {e}")
    print(f"Please check if the ZIP data file exists: {ZIP_DATA}")
    exit(1)

# Join zip data
print("Joining ZIP data …")
if isinstance(zip_gdf, gpd.GeoDataFrame) and 'geometry' in zip_gdf.columns:
    bnb_zip = gpd.sjoin(bnb_gdf, zip_gdf, how="left", predicate="within")
else:
    # go back to non-spatial join if geometry isn't available
    print("No geometry data found, attempting direct ZIP code join...")
    bnb_zip = bnb_gdf.merge(zip_gdf[['zip']], how="left", left_on="neighbourhood", right_on="zip")

# Verify we have zip codes
if bnb_zip['zip'].isna().all():
    print("Warning: No ZIP codes were joined to the listings")

# Load + join school district data
print("Loading school district data …")
try:
    dist_df = pd.read_csv(DIST_DATA)
    # Clean up district data, some columns might have leading/trailing whitespace
    dist_df = dist_df.apply(lambda x: x.str.strip() if isinstance(x, pd.Series) and x.dtype == "object" else x)
    
    # Just keep the district name + location data
    dist_df = dist_df[["NAME", "INTPTLAT", "INTPTLON"]].copy()
    
    # Convert lat/lon to numeric, drop invalid values
    dist_df["INTPTLAT"] = pd.to_numeric(dist_df["INTPTLAT"], errors="coerce")
    dist_df["INTPTLON"] = pd.to_numeric(dist_df["INTPTLON"], errors="coerce")
    dist_df = dist_df.dropna()
    
    # Create geodataframe with the centroid points
    dist_gdf = gpd.GeoDataFrame(
        dist_df,
        geometry=gpd.points_from_xy(dist_df.INTPTLON, dist_df.INTPTLAT),
        crs="EPSG:4326"
    )
    
    # Project to a local utm zone for (somewhat) accurate distance calculations
    # SF is in utm zone 10N (EPSG:32610)
    dist_gdf = dist_gdf.to_crs("EPSG:32610")
    bnb_points = gpd.GeoDataFrame(
        bnb_zip,
        geometry=gpd.points_from_xy(bnb_zip.longitude, bnb_zip.latitude),
        crs="EPSG:4326"
    ).to_crs("EPSG:32610")
    
    # Find nearest school district for each airbnb listing
    print("Finding nearest school district for each listing...")
    nearest_dist = []
    for idx, row in bnb_points.iterrows():
        distances = dist_gdf.geometry.distance(row.geometry)
        nearest_idx = distances.idxmin()
        nearest_dist.append(dist_gdf.iloc[nearest_idx]["NAME"])
    
    bnb_dist = bnb_zip.copy()
    bnb_dist["NAME"] = nearest_dist
    
except Exception as e:
    print(f"Error loading district data: {e}")
    print(f"Please check if the district data file exists: {DIST_DATA}")
    exit(1)

# Calculate nearby school poverty estimate (balltree haversine)
print("Computing school‑poverty radius features …")
poverty = pd.read_csv(POVERTY_CSV, low_memory=False,
                      usecols=["LAT","LON","IPR_EST"]).dropna()
tree   = BallTree(np.radians(poverty[["LAT","LON"]]), metric="haversine")
poverty_vals = poverty["IPR_EST"].values

def avg_poverty(lat, lon, km=POVERTY_RADIUS_KM):
    idx, _ = tree.query_radius(
        np.radians([[lat, lon]]), r=km/6371, return_distance=True
    )
    return float(np.mean(poverty_vals[idx[0]])) if idx[0].size else np.nan

bnb_dist["school_poverty"] = bnb_dist.apply(
    lambda r: avg_poverty(r.latitude, r.longitude), axis=1
)

# Load and process parks/rec facilities data
print("Processing parks and recreation facilities...")
try:
    parks_df = pd.read_csv(PARKS_CSV)
    # Convert parks data to geodataframe
    parks_gdf = gpd.GeoDataFrame(
        parks_df,
        geometry=gpd.points_from_xy(parks_df.longitude, parks_df.latitude),
        crs="EPSG:4326"
    ).to_crs("EPSG:32610")
    
    # Count nearby facilities for each airbnb listing
    print("Counting nearby parks/facilities for each listing...")
    bnb_points = bnb_points.to_crs("EPSG:32610")  # Ensure same projection
    facility_counts = []
    
    for idx, row in bnb_points.iterrows():
        # Count facilities within PARKS_RADIUS_KM
        nearby = parks_gdf[parks_gdf.geometry.distance(row.geometry) <= PARKS_RADIUS_KM * 1000]
        facility_counts.append(len(nearby))
    
    bnb_dist["nearby_facilities"] = facility_counts
    
except Exception as e:
    print(f"Error processing parks data: {e}")
    print(f"Please check if the parks data file exists: {PARKS_CSV}")
    bnb_dist["nearby_facilities"] = np.nan

# Assemble final df
print("Preparing final output...")
try:
    print("Available columns:", bnb_dist.columns.tolist())
    
    # Create output df with column selection + renaming
    column_mapping = {
        'id_left': 'listing_id',
        'neighbourhood': 'location',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'zip': 'zip',
        'NAME': 'district_name',
        'price': 'nightly_price',
        'school_poverty': 'school_poverty',
        'nearby_facilities': 'nearby_facilities',
        'name': 'notes'
    }
    
    # Select only columns we want and rename them
    #Make sure all columns we want exist
    missing_cols = [col for col in column_mapping.keys() if col not in bnb_dist.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    out = bnb_dist[list(column_mapping.keys())].rename(columns=column_mapping)
    
    # Write csv
    print(f"Creating output directory: {OUT_CSV.parent}")
    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    
    print(f"Writing {len(out)} rows to CSV...")
    out.to_csv(OUT_CSV, index=False)
    print(f"Successfully wrote data to {OUT_CSV}")
    
    # Print a sample of the output for verification
    print("\nFirst few rows of output:")
    print(out.head())
    
    # Check output columns
    print("\nOutput columns:")
    print(out.columns.tolist())
    print("\nSample of data types:")
    print(out.dtypes)

except Exception as e:
    print(f"Error preparing output: {e}")
    print("Available columns:", bnb_dist.columns.tolist())
    exit(1)


# CSV columns:

# listing_id            unique airbnb id (string)
# location              airbnb neighbourhood name
# latitude, longitude
# zip                   5‑digit zip (from data)
# district_name         school district name
# nightly_price         cleaned float ($ per night)
# school_poverty        % students below 185 % poverty (float, may be NaN)
# nearby_facilities     # of parks/rec facilities within 1km (integer)
# notes                 airbnb listing title (<= 100 chars)
