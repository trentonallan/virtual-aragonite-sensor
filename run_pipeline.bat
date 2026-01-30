@echo off
REM Virtual Aragonite Sensor - Full Training Pipeline
REM Runs all data processing and model training steps

echo ============================================
echo Virtual Aragonite Sensor Training Pipeline
echo ============================================
echo.

REM Create directories
echo Creating directories...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist models mkdir models
if not exist results mkdir results

REM Step 1: Download GLODAP data
echo.
echo Step 1/4: Processing GLODAP ocean chemistry data...
python scripts\01_download_glodap.py
if errorlevel 1 goto error

REM Step 2: Add bathymetry
echo.
echo Step 2/4: Adding bathymetry data...
python scripts\02_download_bathymetry.py
if errorlevel 1 goto error

REM Step 3: Extract satellite data
echo.
echo Step 3/4: Extracting satellite features from Google Earth Engine...
python scripts\03_extract_satellite_gee.py
if errorlevel 1 goto error

REM Step 4: Train model
echo.
echo Step 4/4: Training model and generating visualizations...
python scripts\04_train_model.py
if errorlevel 1 goto error

echo.
echo ============================================
echo Pipeline complete!
echo ============================================
echo Model saved to: models\aragonite_model.pkl
echo Visualizations saved to: results\
pause
goto end

:error
echo.
echo ============================================
echo Error: Pipeline failed
echo ============================================
pause
exit /b 1

:end