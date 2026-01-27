import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# Initialize Google Earth Engine
print("Initializing Google Earth Engine...")

try:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print("Successfully initialized GEE")

except Exception as e1:
    # If high volume endpoint fails, try standard endpoint
    try:
        ee.Initialize()
        print("Successfully initialized GEE")

    except Exception as e2:
        print(f"Error: {e2}")
        exit(1)

# Load GLODAP surface data
print("\nLoading GLODAP surface data...")
df = pd.read_csv("../data/processed/glodap_with_bathymetry.csv")

# Convert datetime column from string to datetime objects
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"Successfully loaded {len(df):,} samples")

# Define function for extracting data at a single point
def extract_satellite_point(lat, lon, date, days_window=7):
    """Extract MODIS SST and Chlorophyll-a"""

    # Convert dates to strings for Earth Engine
    start_date = (date - timedelta(days=days_window)).strftime('%Y-%m-%d')
    end_date = (date + timedelta(days=days_window)).strftime('%Y-%m-%d')

    # Create Earth Engine geometry object for this location
    point = ee.Geometry.Point([lon, lat])

    # Initalize results dictionary
    # 'modis_sst' - sea surface temperature in degrees Celsius
    # 'modis_chlora' - chlorophyll-a concentration in mg/m^3 
    results = {'modis_sst': np.nan, 'modis_chlora': np.nan}

    try:
        # Access MODIS-Aqua ocean color dataset
        modis = ee.ImageCollection('NASA/OCEANDATA/MODIS-Aqua/L3SMI') \
        .filterDate(start_date, end_date) \
        .filterBounds(point)

        if modis.size().getInfo() > 0:
            sst_image = modis.select('sst').mean()
            sst_value = sst_image.reduceRegion(
                reducer = ee.Reducer.mean(),
                geometry=point,
                scale=4000
            ).getInfo()

            # Store SST and handle None values from GEE
            sst = sst_value.get('sst')
            results['modis_sst'] = float(sst) if sst is not None else np.nan
            
            # Extract chlorophyll-a band and average
            chl_image = modis.select('chlor_a').mean()
            chl_value = chl_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=4000
            ).getInfo()
            
            # Store chlor-a and handle None values from GEE
            chl = chl_value.get('chlor_a')
            results['modis_chlora'] = float(chl) if chl is not None else np.nan
    except:
        pass
    
    return results

# Test extraction
print("\nTesting extraction on first point...")

test_row = df.iloc[0]
print(f"Location: ({test_row['latitude']:.2f}, {test_row['longitude']:.2f})")
print(f"Date: {test_row['datetime'].date()}")

test_result = extract_satellite_point(
    test_row['latitude'],
    test_row['longitude'],
    test_row['datetime']
)
print(f"SST: {test_result['modis_sst']}")
print(f"Chlor-a: {test_result['modis_chlora']}")

response = input(f"\nExtract for all {len(df):,} points? (y/n): ")
if response.lower() != 'y':
    exit(0)

# Extract all satellite data
print("\nExtracting satellite data...")
print(f"This may take between 10 and 60 minutes")
print("="*60)

# Initialize new SST and Chlor-a columns in dataframe
df['modis_sst'] = np.nan
df['modis_chlora'] = np.nan

# Loop through all dataframe rows and display progress bar
start_time = time.time()
successful = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Progress"):
    sat_data = extract_satellite_point(
        row['latitude'], row['longitude'], row['datetime']
    )
    # Store data in dataframe
    df.loc[idx, 'modis_sst'] = sat_data['modis_sst']
    df.loc[idx, 'modis_chlora'] = sat_data['modis_chlora']

    # Check if valid (not NaN)
    if pd.notna(sat_data['modis_sst']) and pd.notna(sat_data['modis_chlora']):
        successful += 1

    # Checkpoint every 100 samples
    if (idx + 1) % 100 == 0:
        df.to_csv("../data/processed/glodap_with_satellite_checkpoint.csv", index=False)
        elapsed = time.time() - start_time
        rate = (idx+1) / elapsed
        eta = (len(df) - idx - 1) / rate / 60
        print(f"\n[{idx+1}/{len(df)}] Success: {successful} | ETA: {eta:.1f}min")

    # Rate limiting
    time.sleep(0.15)

# Save complete dataset
df.to_csv("../data/processed/complete_dataset.csv", index=False)
print(f" Successfully saved as: data/processed/complete_dataset.csv")

# Display statistics
sst_count = df['modis_sst'].notna().sum()
chl_count = df['modis_chlora'].notna().sum()
both_count = (df['modis_sst'].notna() & df['modis_chlora'].notna()).sum()

print(f"\n{'='*60}")
print("Successfully extracted satellite data")
print(f"{'='*60}")
print(f"SST available: {sst_count:,} ({100*sst_count/len(df):.1f}%)")
print(f"Chlor-a available: {chl_count:,} ({100*chl_count/len(df):.1f}%)")
print(f"Both available: {both_count:,} ({100*both_count/len(df):.1f}%)")

# Show sample
if both_count > 0:
    print("\nSample data:")
    cols = ['latitude', 'longitude', 'modis_sst', 'modis_chlora', 'aragonite']
    df_sample = df.dropna(subset=['modis_sst', 'modis_chlora'])
    print(df_sample[cols].head(10).to_string(index=False))
