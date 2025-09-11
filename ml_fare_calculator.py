"""
ML-based taxi fare calculation for Singapore.
Uses trained Random Forest and XGBoost models for predictions.
"""

import pickle
import os
from datetime import datetime
import pandas as pd
import numpy as np

class MLFareCalculator:
    """
    Machine Learning-based fare calculator using trained models.
    """
    
    def __init__(self, models_path='models/'):
        self.models_path = models_path
        self.rf_model = None
        self.xgb_model = None
        self.feature_columns = None
        self.is_loaded = False
        
        # Try to load models if they exist
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk."""
        try:
            # Load Random Forest model
            with open(f'{self.models_path}random_forest_model.pkl', 'rb') as f:
                self.rf_model = pickle.load(f)
            
            # Load XGBoost model
            with open(f'{self.models_path}xgboost_model.pkl', 'rb') as f:
                self.xgb_model = pickle.load(f)
            
            # Load feature columns
            with open(f'{self.models_path}feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            self.is_loaded = True
            print("ML models loaded successfully!")
            
        except FileNotFoundError:
            print("Warning: ML models not found. Please train models first using ml_models.py")
            self.is_loaded = False
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_loaded = False
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula."""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        return c * r
    
    def estimate_duration(self, distance_km):
        """
        Estimate trip duration based on distance.
        Assumes average speed of 30 km/h in Singapore.
        """
        speed_kmh = 30
        time_hours = distance_km / speed_kmh
        time_minutes = time_hours * 60
        return round(time_minutes, 1)
    
    def is_peak_hour(self, hour):
        """Check if given hour is peak hour."""
        return (7 <= hour <= 9) or (18 <= hour <= 20)
    
    def is_weekend(self, day_of_week):
        """Check if given day is weekend (0=Monday, 6=Sunday)."""
        return day_of_week in [5, 6]  # Saturday, Sunday
    
    def predict_fare_ml(self, distance_km, duration_minutes, passengers=1, 
                       hour=None, day_of_week=None, month=None, 
                       is_peak_hour=None, is_weekend=None):
        """
        Predict fare using ML models.
        
        Args:
            distance_km: Trip distance in kilometers
            duration_minutes: Trip duration in minutes
            passengers: Number of passengers (1-4)
            hour: Hour of day (0-23), if None uses current hour
            day_of_week: Day of week (0=Monday, 6=Sunday), if None uses current day
            month: Month (1-12), if None uses current month
            is_peak_hour: Whether it's peak hour, if None calculates automatically
            is_weekend: Whether it's weekend, if None calculates automatically
        
        Returns:
            Dictionary with predictions from both models
        """
        if not self.is_loaded:
            raise ValueError("ML models not loaded. Please train models first.")
        
        # Use current time if not provided
        now = datetime.now()
        if hour is None:
            hour = now.hour
        if day_of_week is None:
            day_of_week = now.weekday()
        if month is None:
            month = now.month
        if is_peak_hour is None:
            is_peak_hour = self.is_peak_hour(hour)
        if is_weekend is None:
            is_weekend = self.is_weekend(day_of_week)
        
        # Create feature vector
        features = pd.DataFrame({
            'distance_km': [distance_km],
            'duration_minutes': [duration_minutes],
            'passengers': [passengers],
            'hour': [hour],
            'day_of_week': [day_of_week],
            'month': [month],
            'is_peak_hour': [int(is_peak_hour)],
            'is_weekend': [int(is_weekend)],
            'distance_squared': [distance_km ** 2],
            'duration_squared': [duration_minutes ** 2],
            'distance_duration_ratio': [distance_km / (duration_minutes + 1e-6)],
            'is_morning': [int(6 <= hour < 12)],
            'is_afternoon': [int(12 <= hour < 18)],
            'is_evening': [int(18 <= hour < 24)],
            'is_night': [int(0 <= hour < 6)],
            'is_rainy_season': [int((month >= 11) or (month <= 3))]
        })
        
        # Ensure correct column order
        features = features[self.feature_columns]
        
        # Make predictions
        rf_pred = self.rf_model.predict(features)[0]
        xgb_pred = self.xgb_model.predict(features)[0]
        
        return {
            'Random Forest': round(rf_pred, 2),
            'XGBoost': round(xgb_pred, 2),
            'Average': round((rf_pred + xgb_pred) / 2, 2)
        }
    
    def predict_fare_simple(self, distance_km, duration_minutes, passengers=1, 
                           hour=None, day_of_week=None, month=None):
        """
        Fallback to simple formula if ML models are not available.
        """
        # Use current time if not provided
        now = datetime.now()
        if hour is None:
            hour = now.hour
        if day_of_week is None:
            day_of_week = now.weekday()
        
        # Calculate base fare
        base_fare = 3.20
        distance_cost = 1.50 * distance_km
        time_cost = 0.40 * duration_minutes
        passenger_cost = (passengers - 1) * 0.50
        
        total_fare = base_fare + distance_cost + time_cost + passenger_cost
        
        # Apply surcharges
        if self.is_peak_hour(hour):
            total_fare *= 1.20
        if self.is_weekend(day_of_week):
            total_fare *= 1.10
        
        return round(total_fare, 2)
    
    def calculate_fare(self, distance_km, duration_minutes, passengers=1, 
                      hour=None, day_of_week=None, month=None):
        """
        Main method to calculate fare using ML models or fallback to simple formula.
        """
        if self.is_loaded:
            try:
                return self.predict_fare_ml(
                    distance_km, duration_minutes, passengers,
                    hour, day_of_week, month
                )
            except Exception as e:
                print(f"ML prediction failed: {e}")
                print("Falling back to simple formula...")
        
        # Fallback to simple formula
        simple_fare = self.predict_fare_simple(
            distance_km, duration_minutes, passengers,
            hour, day_of_week, month
        )
        
        return {
            'Simple Formula': simple_fare,
            'Note': 'ML models not available, using simple formula'
        }


def main():
    """Test the ML fare calculator."""
    print("Testing ML Fare Calculator")
    print("=" * 30)
    
    # Initialize calculator
    calculator = MLFareCalculator()
    
    # Test prediction
    test_cases = [
        {
            'distance_km': 5.0,
            'duration_minutes': 15.0,
            'passengers': 1,
            'description': 'Normal trip'
        },
        {
            'distance_km': 10.0,
            'duration_minutes': 25.0,
            'passengers': 2,
            'description': 'Longer trip with 2 passengers'
        },
        {
            'distance_km': 3.0,
            'duration_minutes': 8.0,
            'passengers': 1,
            'hour': 8,  # Peak hour
            'description': 'Peak hour trip'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case['description']}")
        print("-" * 40)
        
        result = calculator.calculate_fare(
            distance_km=case['distance_km'],
            duration_minutes=case['duration_minutes'],
            passengers=case['passengers'],
            hour=case.get('hour')
        )
        
        for model, prediction in result.items():
            if isinstance(prediction, (int, float)):
                print(f"{model}: ${prediction} SGD")
            else:
                print(f"{model}: {prediction}")


if __name__ == "__main__":
    main()
