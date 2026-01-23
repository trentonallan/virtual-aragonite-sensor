import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import time

print("Initializing Google Earth Engine...")

try:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print("✓ GEE initialized successfully")
except Exception as e1:
    try:
        ee.Initialize()
        print("✓ GEE initialized successfully")
    except Exception as e2:
        print(f"Error: {e2}")
        exit(1)

print("\nLoading GLODAP surface data...")
df = pd.read_csv("../data/processed/glodap_with_bathymetry.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"✓ Loaded {len(df):,} samples")

def extract_satellite_point(lat, lon, date, days_window=7):
    """Extract MODIS SST and Chlorophyll-a"""
    start_date = (date - timedelta(days=days_window)).strftime('%Y-%m-%d')
    end_date = (date + timedelta(days=days_window)).strftime('%Y-%m-%d')
    
    point = ee.Geometry.Point([lon, lat])
    results = {'modis_sst': np.nan, 'modis_chlora': np.nan}
    
    try:
        modis = ee.ImageCollection('NASA/OCEANDATA/MODIS-Aqua/L3SMI') \
            .filterDate(start_date, end_date) \
            .filterBounds(point)
        
        if modis.size().getInfo() > 0:
            sst_image = modis.select('sst').mean()
            sst_value = sst_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=4000
            ).getInfo()
            
            # Handle None values from GEE
            sst = sst_value.get('sst')
            results['modis_sst'] = float(sst) if sst is not None else np.nan
            
            chl_image = modis.select('chlor_a').mean()
            chl_value = chl_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=4000
            ).getInfo()
            
            chl = chl_value.get('chlor_a')
            results['modis_chlora'] = float(chl) if chl is not None else np.nan
    except:
        pass
    
    return results

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

print("\nExtracting satellite data...")
print(f"Estimated time: ~10-15 minutes")
print("="*60)

df['modis_sst'] = np.nan
df['modis_chlora'] = np.nan

start_time = time.time()
successful = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Progress"):
    sat_data = extract_satellite_point(
        row['latitude'], row['longitude'], row['datetime']
    )
    df.loc[idx, 'modis_sst'] = sat_data['modis_sst']
    df.loc[idx, 'modis_chlora'] = sat_data['modis_chlora']
    
    # Check if valid (not NaN)
    if pd.notna(sat_data['modis_sst']):
        successful += 1
    
    if (idx + 1) % 100 == 0:
        df.to_csv("../data/processed/glodap_with_satellite_checkpoint.csv", index=False)
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed
        eta = (len(df) - idx - 1) / rate / 60
        print(f"\n  [{idx+1}/{len(df)}] Success: {successful} | ETA: {eta:.1f}min")
    
    time.sleep(0.15)

df.to_csv("../data/processed/glodap_with_satellite.csv", index=False)

# Also save as complete_dataset for training pipeline
df.to_csv("../data/processed/complete_dataset.csv", index=False)
print(f"✓ Also saved as: data/processed/complete_dataset.csv")

sst_count = df['modis_sst'].notna().sum()
chl_count = df['modis_chlora'].notna().sum()
both_count = (df['modis_sst'].notna() & df['modis_chlora'].notna()).sum()

print(f"\n{'='*60}")
print("EXTRACTION COMPLETE")
print(f"{'='*60}")
print(f"SST available: {sst_count:,} ({100*sst_count/len(df):.1f}%)")
print(f"Chlor-a available: {chl_count:,} ({100*chl_count/len(df):.1f}%)")
print(f"Both available: {both_count:,} ({100*both_count/len(df):.1f}%)")

df_clean = df.dropna(subset=['modis_sst', 'modis_chlora'])
df_clean.to_csv("../data/processed/glodap_with_bathymetry_and_satellite_clean.csv", index=False)

print(f"\n✓ Clean dataset: {len(df_clean):,} samples")
print(f"✓ Saved: data/processed/glodap_with_satellite_clean.csv")

if len(df_clean) > 0:
    print(f"\n{'='*60}")
    print("✓ PHASE 2 COMPLETE - READY FOR ML TRAINING")
    print(f"{'='*60}")
    print("\nSample data:")
    cols = ['latitude', 'longitude', 'modis_sst', 'modis_chlora', 'aragonite']
    print(df_clean[cols].head(10).to_string(index=False))
