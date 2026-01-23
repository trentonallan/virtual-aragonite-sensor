import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# Ignore pandas warnings caused by GLODAP data containing mixed types
# (data has already been validated)
warnings.filterwarnings('ignore')

print("Starting GLODAP processing...")

# Create directories for data
Path("../data/raw").mkdir(parents=True, exist_ok=True)
Path("../data/processed").mkdir(parents=True, exist_ok=True)

# Define file path to read data from
output_path = "../data/raw/glodap_v2_2023.csv"

print("Loading data...")

# G2latitude - latitude coordinate in decimal degrees [-90, 90] where negative = South and positive = North 
# G2longitude - longitude coordinate in decimal degrees [-180, 180] where negative = West and postiive = East
# G2depth - depth of measurement in meters below sea surface (0 = surface)
# G2year - year the measurement was taken
# G2month - month the measurement was taken [1-12]
# G2day - day the measurement was taken [1-31]
# G2temperature - sea surface temperature in degrees Celsius
# G2salinity - salinity in PSU (Practical Salinity Units)
# G2tco2 - total dissolved inorganic carbon (DIC) in micromole per kilogram
# G2talk - total alkalinity in micromole per kilogram (represents buffering capacity via ability to neutralize acids)
# G2phts25p0 - pH on the total hydrogren ion scale, measured at 25 degrees Celcius and 0 dbar pressure (sea surface)
needed_cols = [
    'G2latitude', 'G2longitude', 'G2depth', 'G2year', 'G2month', 'G2day',
    'G2temperature', 'G2salinity', 'G2tco2', 'G2talk', 'G2phts25p0'
]

# Create DataFrame object
df = pd.read_csv(output_path, usecols=needed_cols, low_memory=False)

print(f"Successfully loaded {len(df):,} records")

# Filter DataFrame object for usability
print("Filtering...")

df_surface = df[
    (df['G2depth'] <= 10) & # Only surface ocean (0-10 meter depth)
    (df['G2temperature'].notna()) & # Cannot calculate aragonite saturation without temperature
    (df['G2salinity'].notna()) & # Determines calcium ion concentration which accurate carbonate chemistry requires 
    (df['G2latitude'].between(-60, 60)) & # Reef-buildings corals exist primarily between 30 degrees N and 30 degrees S
    (df['G2year'] >= 2000) & # Filter out historical data from before year 2000
    (df['G2tco2'].notna()) & # Cannot calculate aragonite saturation without DIC (dissolved inorganic carbon)
    (df['G2talk'].notna()) & # Cannot calculate aragonite saturation without alkalinity 
    (df['G2phts25p0'].notna()) & # Provides independent validation of carbonate calculations
    (df['G2phts25p0'] > 6.0) & # Ocean pH below 6.0 is unrealistic (Ocean pH is generally 7.9-8.4)
    (df['G2phts25p0'] < 9.0) & # Ocean pH above 9.0 is unrealistic 
    (df['G2tco2'] > 100) & # DIC below 100 is unrealistic for surface ocean
    (df['G2tco2'] < 3000) & # DIC above 3000 is unrealistic 
    (df['G2talk'] > 100) & # Alkalinity below 100 is unrealistic for surface ocean
    (df['G2talk'] < 3000) & # Alkalinity above 3000 is unrealistic outside of hypersaline environments
    (df['G2temperature'] > -2) & # Seawater freezes around -1.8 degrees Celcius 
    (df['G2temperature'] < 40) &  # Seawater above 40 degrees Celcius is unrealistic even in extreme shallow tropical environments 
    (df['G2salinity'] > 10) & # Salinity above 10 PSU unrealistic for open ocean
    (df['G2salinity'] < 45) # Allow legitimately salty regions while catching extreme outliers
    ].copy()

print(f"Successfully filtered for {len(df_surface):,} surface samples")

# Randomly sample to reduce dataset sie
df_surface = df_surface.sample(n=min(3000, len(df_surface)), random_state=42)

# Rename columns
df_surface = df_surface.rename(columns={
    'G2latitude': 'latitude',
    'G2longitude': 'longitude',
    'G2depth': 'depth',
    'G2year': 'year',
    'G2month': 'month',
    'G2day': 'day',
    'G2temperature': 'temperature',
    'G2salinity': 'salinity',
    'G2talk': 'alkalinity',
    'G2tco2': 'dic',
    'G2phts25p0': 'ph'
})

# Create datetime column
df_surface['datetime'] = pd.to_datetime(
    df_surface['year'].astype(int).astype(str) + '-' +
    df_surface['month'].astype(int).astype(str).str.zfill(2) + '-' +
    df_surface['day'].astype(int).astype(str).str.zfill(2),
    format='%Y-%m-%d'
)

# Print temporal span of data
print(f"Date range: {df_surface['datetime'].min().date()} to {df_surface['datetime'].max().date()}")

# Omega calculation
# Based on: omega ≈ f(Alk - DIC, temp, salinity)

print("\nCalculating Ωarag using empirical model...")

def calculate_omega(alk, dic, temp, sal):
    """
    Simplified omega calculation.
    Based on the fact that omega correlates strongly with (Alk-DIC).
    """
    # Temperature in Kelvin
    TempK = temp + 273.15
    
    # Excess alkalinity (proxy for CO3)
    excess_alk = alk - dic
    
    # Calcium concentration (mmol/kg)
    ca = 10.28 * (sal / 35.0)
    
    # Temperature correction factor
    temp_factor = np.exp(-0.02 * (temp - 25))
    
    # Empirical relationship
    # Omega ≈ 0.01 * excess_alk * temp_factor
    omega = 0.015 * excess_alk * temp_factor * (sal / 35.0)
    
    return omega

df_surface['aragonite'] = calculate_omega(
    df_surface['alkalinity'].values,
    df_surface['dic'].values,
    df_surface['temperature'].values,
    df_surface['salinity'].values
)

print("\nΩarag statistics:")
print(f"  Min: {df_surface['aragonite'].min():.2f}")
print(f"  Max: {df_surface['aragonite'].max():.2f}")
print(f"  Mean: {df_surface['aragonite'].mean():.2f}")

# Filter DataFrame object for valid calculated aragonite values
initial = len(df_surface)

df_surface = df_surface[
    (df_surface['aragonite'].notna()) & # Aragonite values must be non-null
    (df_surface['aragonite'] > 0.5) & # Catches calculation errors and extreme outliers
    (df_surface['aragonite'] < 8.0) # Aragonite saturation state above 8.0 unrealistic for most reef environments
].copy()

print(f"\nSuccessfully kept {len(df_surface):,} / {initial:,} valid samples")

# Save data to csv file
df_surface.to_csv("../data/processed/glodap_surface.csv", index=False)
print(f"Successfully saved to: data/processed/glodap_surface.csv")

# Print analytics
print("\n" + "="*60)
print("PHASE 1 COMPLETE - GLODAP SURFACE DATA")
print("="*60)
print(f"Total samples: {len(df_surface):,}")
print(f"Date range: {df_surface['datetime'].min().date()} to {df_surface['datetime'].max().date()}")
print(f"\nΩarag: {df_surface['aragonite'].mean():.2f} ± {df_surface['aragonite'].std():.2f}")
print(f"  Range: {df_surface['aragonite'].min():.2f} - {df_surface['aragonite'].max():.2f}")
print(f"Temp: {df_surface['temperature'].mean():.1f}°C ± {df_surface['temperature'].std():.1f}")
print(f"Sal: {df_surface['salinity'].mean():.2f} ± {df_surface['salinity'].std():.2f}")
print(f"pH: {df_surface['ph'].mean():.2f} ± {df_surface['ph'].std():.2f}")

print("\nSample data (first 10):")
print(df_surface[['latitude', 'longitude', 'temperature', 'salinity', 'ph', 'aragonite']].head(10).to_string(index=False))

print("\n" + "="*60)
print("Next step: Extract satellite data for these locations")
print("="*60)