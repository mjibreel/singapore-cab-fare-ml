# Singapore Taxi Fare Prediction

A simple Python application to predict taxi fares in Singapore.


streamlit run streamlit_app.py



## What it does
- Calculates taxi fare based on distance and time using **TWO DIFFERENT PRICING MODELS**
- Uses simple formulas (no complex ML)
- Compares standard vs premium pricing strategies
- Easy to understand and modify

## Files
- `main.py` - Command-line application with two pricing models
- `streamlit_app.py` - Beautiful web application (Streamlit)
- `haversine.py` - Distance calculation using Haversine formula
- `fare_calculator.py` - Fare calculation logic and surcharges
- `requirements.txt` - Python packages needed

## How to run

### Option 1: Command-Line App
1. Install packages: `pip install -r requirements.txt`
2. Run the command-line app: `python main.py`

### Option 2: Streamlit Web App (Recommended)
1. Install packages: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run streamlit_app.py`
3. Open your web browser to the displayed URL



## Two Pricing Models

### Model 1: Standard Pricing
- **Base fare**: $3.20 SGD
- **Distance cost**: $1.50 per kilometer
- **Time cost**: $0.40 per minute
- **Peak hour surcharge**: 20% (7-9 AM, 6-8 PM)
- **Weekend surcharge**: 10% (Saturday, Sunday)

**Formula**: `$3.20 + ($1.50 × distance_km) + ($0.40 × duration_min)`

### Model 2: Premium Pricing
- **Base fare**: $4.00 SGD (higher)
- **Distance cost**: $1.80 per kilometer (higher)
- **Time cost**: $0.35 per minute (lower)
- **Peak hour surcharge**: 25% (higher)
- **Weekend surcharge**: 15% (higher)

**Formula**: `$4.00 + ($1.80 × distance_km) + ($0.35 × duration_min)`

## How it works
1. User enters pickup and dropoff locations (landmarks or custom coordinates)
2. App calculates distance using Haversine formula
3. App estimates trip duration (assumes 30 km/h average speed)
4. App calculates fare using **BOTH MODELS** for comparison
5. Shows detailed breakdown and comparison of both pricing strategies

## Key Features

### Location Input Options
1. **Popular Landmarks**: Pre-defined Singapore locations (Marina Bay Sands, Zoo, Airport, etc.)
2. **Custom Coordinates**: User can enter specific latitude/longitude coordinates

### Fare Calculation Features
- **Real-time calculation** based on current date/time
- **Peak hour detection** (7-9 AM, 6-8 PM)
- **Weekend detection** (Saturday, Sunday)
- **Automatic surcharge application**
- **Detailed fare breakdown** for both models

### Technical Implementation
- **Haversine formula** for accurate distance calculation
- **Modular design** with separate functions for each component
- **Error handling** for invalid inputs
- **User-friendly interface** with clear prompts and examples

## Example Output
```
💰 FARE PREDICTIONS (TWO MODELS)
--------------------------------------------------
Model 1 (Standard): $38.36 SGD
Model 2 (Premium): $42.22 SGD
Average: $40.29 SGD
```

That's it! Simple and straightforward, perfect for a school project report.
# singapore-taxi-fare-prediction
