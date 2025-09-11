"""
Machine Learning Models for Singapore Taxi Fare Prediction
Implements Random Forest and XGBoost as mentioned in the proposal.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import os
from datetime import datetime, timedelta
import random

class TaxiFareMLPredictor:
    """
    Machine Learning-based taxi fare prediction system.
    Implements Random Forest and XGBoost models as per proposal.
    """
    
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.feature_columns = None
        self.is_trained = False
        
    def generate_synthetic_data(self, n_samples=10000):
        """
        Generate synthetic training data based on Singapore taxi patterns.
        This replaces the need for real data as mentioned in the proposal.
        """
        print("Generating synthetic training data...")
        
        # Singapore coordinates bounds
        singapore_bounds = {
            'lat_min': 1.15, 'lat_max': 1.47,
            'lon_min': 103.6, 'lon_max': 104.1
        }
        
        data = []
        
        for _ in range(n_samples):
            # Generate random pickup and dropoff coordinates within Singapore
            pickup_lat = random.uniform(singapore_bounds['lat_min'], singapore_bounds['lat_max'])
            pickup_lon = random.uniform(singapore_bounds['lon_min'], singapore_bounds['lon_max'])
            dropoff_lat = random.uniform(singapore_bounds['lat_min'], singapore_bounds['lat_max'])
            dropoff_lon = random.uniform(singapore_bounds['lon_min'], singapore_bounds['lon_max'])
            
            # Calculate distance using Haversine formula
            distance = self._haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
            
            # Skip very short trips (less than 0.5km)
            if distance < 0.5:
                continue
                
            # Generate random datetime
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 12, 31)
            random_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Calculate features
            hour = random_date.hour
            day_of_week = random_date.weekday()  # 0=Monday, 6=Sunday
            month = random_date.month
            
            # Estimate duration (30 km/h average speed)
            duration = (distance / 30) * 60  # Convert to minutes
            
            # Generate passenger count (1-4 passengers)
            passengers = random.choices([1, 2, 3, 4], weights=[0.6, 0.25, 0.1, 0.05])[0]
            
            # Determine if peak hour
            is_peak_hour = (7 <= hour <= 9) or (18 <= hour <= 20)
            
            # Determine if weekend
            is_weekend = day_of_week in [5, 6]  # Saturday, Sunday
            
            # Generate realistic fare using Singapore taxi pricing
            base_fare = 3.20
            distance_cost = 1.50 * distance
            time_cost = 0.40 * duration
            passenger_cost = (passengers - 1) * 0.50  # Additional cost per extra passenger
            
            # Base fare calculation
            fare = base_fare + distance_cost + time_cost + passenger_cost
            
            # Apply surcharges
            if is_peak_hour:
                fare *= 1.20  # 20% peak hour surcharge
            if is_weekend:
                fare *= 1.10  # 10% weekend surcharge
                
            # Add some realistic noise/variation
            noise_factor = random.uniform(0.95, 1.05)
            fare *= noise_factor
            
            # Round to 2 decimal places
            fare = round(fare, 2)
            
            data.append({
                'pickup_lat': pickup_lat,
                'pickup_lon': pickup_lon,
                'dropoff_lat': dropoff_lat,
                'dropoff_lon': dropoff_lon,
                'distance_km': round(distance, 2),
                'duration_minutes': round(duration, 1),
                'passengers': passengers,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'is_peak_hour': int(is_peak_hour),
                'is_weekend': int(is_weekend),
                'fare_sgd': fare
            })
        
        return pd.DataFrame(data)
    
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
    
    def prepare_features(self, df):
        """Prepare features for machine learning."""
        # Create additional features
        df['distance_squared'] = df['distance_km'] ** 2
        df['duration_squared'] = df['duration_minutes'] ** 2
        df['distance_duration_ratio'] = df['distance_km'] / (df['duration_minutes'] + 1e-6)
        
        # Time-based features
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
        
        # Seasonal features
        df['is_rainy_season'] = ((df['month'] >= 11) | (df['month'] <= 3)).astype(int)
        
        # Feature columns for training
        self.feature_columns = [
            'distance_km', 'duration_minutes', 'passengers',
            'hour', 'day_of_week', 'month',
            'is_peak_hour', 'is_weekend',
            'distance_squared', 'duration_squared', 'distance_duration_ratio',
            'is_morning', 'is_afternoon', 'is_evening', 'is_night',
            'is_rainy_season'
        ]
        
        return df[self.feature_columns], df['fare_sgd']
    
    def train_models(self, X, y):
        """Train both Random Forest and XGBoost models."""
        print("Training Random Forest model...")
        
        # Random Forest with hyperparameter tuning
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        rf_grid.fit(X, y)
        self.rf_model = rf_grid.best_estimator_
        
        print("Training XGBoost model...")
        
        # XGBoost with hyperparameter tuning
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_grid = GridSearchCV(
            xgb.XGBRegressor(random_state=42),
            xgb_param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        xgb_grid.fit(X, y)
        self.xgb_model = xgb_grid.best_estimator_
        
        self.is_trained = True
        print("Model training completed!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate both models and return performance metrics."""
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation!")
        
        results = {}
        
        # Random Forest evaluation
        rf_pred = self.rf_model.predict(X_test)
        results['Random Forest'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'MAE': mean_absolute_error(y_test, rf_pred),
            'R²': r2_score(y_test, rf_pred)
        }
        
        # XGBoost evaluation
        xgb_pred = self.xgb_model.predict(X_test)
        results['XGBoost'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'MAE': mean_absolute_error(y_test, xgb_pred),
            'R²': r2_score(y_test, xgb_pred)
        }
        
        return results
    
    def predict_fare(self, distance_km, duration_minutes, passengers=1, 
                    hour=12, day_of_week=0, month=6, is_peak_hour=False, is_weekend=False):
        """
        Predict fare using both models.
        Returns predictions from both Random Forest and XGBoost.
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions!")
        
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
    
    def save_models(self, filepath_prefix='models/'):
        """Save trained models to disk."""
        if not self.is_trained:
            raise ValueError("Models must be trained before saving!")
        
        # Create models directory if it doesn't exist
        os.makedirs(filepath_prefix, exist_ok=True)
        
        # Save Random Forest model
        with open(f'{filepath_prefix}random_forest_model.pkl', 'wb') as f:
            pickle.dump(self.rf_model, f)
        
        # Save XGBoost model
        with open(f'{filepath_prefix}xgboost_model.pkl', 'wb') as f:
            pickle.dump(self.xgb_model, f)
        
        # Save feature columns
        with open(f'{filepath_prefix}feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print(f"Models saved to {filepath_prefix}")
    
    def load_models(self, filepath_prefix='models/'):
        """Load trained models from disk."""
        # Load Random Forest model
        with open(f'{filepath_prefix}random_forest_model.pkl', 'rb') as f:
            self.rf_model = pickle.load(f)
        
        # Load XGBoost model
        with open(f'{filepath_prefix}xgboost_model.pkl', 'rb') as f:
            self.xgb_model = pickle.load(f)
        
        # Load feature columns
        with open(f'{filepath_prefix}feature_columns.pkl', 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        self.is_trained = True
        print("Models loaded successfully!")


def main():
    """Main function to train and save models."""
    print("Singapore Taxi Fare Prediction - ML Model Training")
    print("=" * 50)
    
    # Initialize predictor
    predictor = TaxiFareMLPredictor()
    
    # Generate synthetic training data
    print("Generating training data...")
    df = predictor.generate_synthetic_data(n_samples=15000)
    print(f"Generated {len(df)} training samples")
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    predictor.train_models(X_train, y_train)
    
    # Evaluate models
    print("\nModel Evaluation Results:")
    print("-" * 30)
    results = predictor.evaluate_models(X_test, y_test)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save models
    predictor.save_models()
    
    # Test prediction
    print("\nTest Prediction:")
    print("-" * 20)
    test_pred = predictor.predict_fare(
        distance_km=5.0,
        duration_minutes=15.0,
        passengers=2,
        hour=14,
        day_of_week=1,
        month=6,
        is_peak_hour=False,
        is_weekend=False
    )
    
    for model, prediction in test_pred.items():
        print(f"{model}: ${prediction} SGD")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
