#!/usr/bin/env python3
"""
Training script for Singapore Taxi Fare Prediction ML models.
Run this script to train and save the Random Forest and XGBoost models.
"""

import sys
import os
from ml_models import TaxiFareMLPredictor

def main():
    """Main training function."""
    print("🚕 Singapore Taxi Fare Prediction - Model Training")
    print("=" * 60)
    print()
    
    # Check if required packages are installed
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost as xgb
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return
    
    print()
    
    # Initialize predictor
    predictor = TaxiFareMLPredictor()
    
    # Generate synthetic training data
    print("📊 Generating synthetic training data...")
    print("   This creates realistic Singapore taxi trip data for training")
    df = predictor.generate_synthetic_data(n_samples=15000)
    print(f"   ✅ Generated {len(df)} training samples")
    print()
    
    # Show sample of data
    print("📋 Sample of generated data:")
    print(df.head())
    print()
    
    # Prepare features
    print("🔧 Preparing features for machine learning...")
    X, y = predictor.prepare_features(df)
    print(f"   ✅ Features shape: {X.shape}")
    print(f"   ✅ Target shape: {y.shape}")
    print(f"   ✅ Feature columns: {len(predictor.feature_columns)}")
    print()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("📊 Data split:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print()
    
    # Train models
    print("🤖 Training machine learning models...")
    print("   This may take a few minutes...")
    print()
    
    predictor.train_models(X_train, y_train)
    print("   ✅ Model training completed!")
    print()
    
    # Evaluate models
    print("📈 Evaluating model performance...")
    results = predictor.evaluate_models(X_test, y_test)
    
    print("\n🎯 Model Performance Results:")
    print("=" * 40)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"   RMSE: ${metrics['RMSE']:.2f} SGD")
        print(f"   MAE:  ${metrics['MAE']:.2f} SGD")
        print(f"   R²:   {metrics['R²']:.4f}")
    
    # Check if models meet the proposal target (RMSE < $3.00)
    print("\n🎯 Performance Check:")
    print("-" * 20)
    for model_name, metrics in results.items():
        if metrics['RMSE'] < 3.0:
            print(f"✅ {model_name}: RMSE ${metrics['RMSE']:.2f} < $3.00 (Target met!)")
        else:
            print(f"⚠️  {model_name}: RMSE ${metrics['RMSE']:.2f} >= $3.00 (Target not met)")
    
    # Save models
    print("\n💾 Saving trained models...")
    predictor.save_models()
    print("   ✅ Models saved successfully!")
    print()
    
    # Test prediction
    print("🧪 Testing model predictions...")
    print("-" * 35)
    
    test_cases = [
        {
            'distance_km': 5.0,
            'duration_minutes': 15.0,
            'passengers': 1,
            'description': 'Marina Bay to Sentosa (Normal hours)'
        },
        {
            'distance_km': 10.0,
            'duration_minutes': 25.0,
            'passengers': 2,
            'description': 'Airport to City Center (2 passengers)'
        },
        {
            'distance_km': 3.0,
            'duration_minutes': 8.0,
            'passengers': 1,
            'hour': 8,
            'description': 'Short trip during peak hours'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['description']}")
        print(f"Distance: {case['distance_km']} km, Duration: {case['duration_minutes']} min")
        
        test_pred = predictor.predict_fare(
            distance_km=case['distance_km'],
            duration_minutes=case['duration_minutes'],
            passengers=case['passengers'],
            hour=case.get('hour', 12),
            day_of_week=1,
            month=6,
            is_peak_hour=case.get('hour', 12) in [8, 19],
            is_weekend=False
        )
        
        for model, prediction in test_pred.items():
            print(f"   {model}: ${prediction} SGD")
    
    print("\n🎉 Training completed successfully!")
    print("\nNext steps:")
    print("1. Run your Streamlit app: streamlit run streamlit_app.py")
    print("2. The app will now use ML models for predictions!")
    print("3. Check the 'models/' folder for saved model files")

if __name__ == "__main__":
    main()
