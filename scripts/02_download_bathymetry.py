import pandas as pd
import numpy as np
import xarray as xr
import requests
import gzip
import shutil
from pathlib import Path

print("="*60)
print("ADDING BATHYMETRY DATA")
print("="*60)

# Load your data
df = pd.read_csv("../data/processed/glodap_surface.csv")
print(f"\nLoaded {len(df):,} samples")

# Download ETOPO1 (ice surface, grid-registered)
# This is ~400MB - will take a few minutes
data_dir = Path("../data/raw")
bathy_file = data_dir / "ETOPO1_Ice_g_gmt4.grd"

if not bathy_file.exists():
    print("\nDownloading ETOPO1 bathymetry (~400MB)...")
    print("This may take 5-10 minutes...")
    
    url = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/netcdf/ETOPO1_Ice_g_gmt4.grd.gz"
    
    # Download compressed file
    gz_file = data_dir / "ETOPO1_Ice_g_gmt4.grd.gz"
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(gz_file, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    print("\n\nDecompressing...")
    with gzip.open(gz_file, 'rb') as f_in:
        with open(bathy_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Clean up
    gz_file.unlink()
    print("✓ Download complete")
else:
    print("\n✓ Bathymetry file already exists")

# Load bathymetry dataset
print("\nLoading bathymetry grid...")
bathy = xr.open_dataset(bathy_file)
print("✓ Loaded")

# Extract depth at each location
print("\nExtracting bathymetry for each sample...")

bathymetry_values = []

for idx, row in df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    
    # ETOPO1 uses longitude range -180 to 180
    # Make sure longitude is in correct range
    if lon > 180:
        lon = lon - 360
    
    try:
        # Get nearest bathymetry value
        depth = float(bathy.sel(x=lon, y=lat, method='nearest')['z'].values)
        
        # Negative values = below sea level
        # Convert to positive depth
        seafloor_depth = -depth if depth < 0 else 0
        
        bathymetry_values.append(seafloor_depth)
    except:
        # If outside grid, use NaN
        bathymetry_values.append(np.nan)
    
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/{len(df)} samples", end='\r')

print(f"\n✓ Extracted bathymetry for all samples")

# Add to dataframe
df['bathymetry_m'] = bathymetry_values

# Check statistics
print("\n" + "="*60)
print("BATHYMETRY STATISTICS")
print("="*60)
print(f"Mean depth: {df['bathymetry_m'].mean():.0f} m")
print(f"Median depth: {df['bathymetry_m'].median():.0f} m")
print(f"Min depth: {df['bathymetry_m'].min():.0f} m")
print(f"Max depth: {df['bathymetry_m'].max():.0f} m")
print(f"Missing values: {df['bathymetry_m'].isna().sum()}")

# Depth distribution
print("\nDepth distribution:")
bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
for i in range(len(bins)-1):
    count = ((df['bathymetry_m'] >= bins[i]) & (df['bathymetry_m'] < bins[i+1])).sum()
    print(f"  {bins[i]:5d}-{bins[i+1]:5d}m: {count:4d} samples ({100*count/len(df):5.1f}%)")

# Save updated dataset
output_path = "../data/processed/glodap_with_bathymetry.csv"
df.to_csv(output_path, index=False)
print(f"\n✓ Saved: {output_path}")

# Show sample
print("\n" + "="*60)
print("SAMPLE DATA (with bathymetry)")
print("="*60)
cols = ['latitude', 'longitude', 'bathymetry_m', 'temperature', 'aragonite']
print(df[cols].head(10).to_string(index=False))

print("\n✓ Ready to retrain model with bathymetry feature!")