"""
Singapore Taxi Fare Prediction - ML-Enhanced Streamlit Web Application
Beautiful web interface with Machine Learning models for taxi fare predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Import our utility functions
from haversine import calculate_distance, get_singapore_landmarks
from fare_calculator import calculate_fare, estimate_duration, is_peak_hour, is_weekend
from ml_fare_calculator import MLFareCalculator

# Page configuration
st.set_page_config(
    page_title="Singapore Taxi Fare Predictor - ML Edition",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .ml-prediction-box {
        background-color: #e8f5e8;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
    .info-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🚕 Singapore Taxi Fare Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        Predict taxi fares in Singapore using <strong>Machine Learning</strong> models
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize ML calculator
    ml_calculator = MLFareCalculator()
    
    # Check if ML models are available
    ml_available = ml_calculator.is_loaded
    
    if not ml_available:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ ML Models Not Found</h4>
            <p>Machine Learning models are not available. Please train the models first:</p>
            <code>python train_models.py</code>
            <p>The app will use simple formula calculations as fallback.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            <h4>✅ ML Models Loaded</h4>
            <p>Using trained Random Forest and XGBoost models for predictions!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar for input
    with st.sidebar:
        st.markdown("## 🎯 Trip Details")
        
        # Location input method
        input_method = st.radio(
            "Choose input method:",
            ["Popular Landmarks", "Custom Coordinates"],
            index=0
        )
        
        landmarks = get_singapore_landmarks()
        
        if input_method == "Popular Landmarks":
            # Pickup location
            pickup_choice = st.selectbox(
                "📍 Pickup Location:",
                list(landmarks.keys()),
                index=0
            )
            pickup_coords = landmarks[pickup_choice]
            
            # Dropoff location
            dropoff_choice = st.selectbox(
                "🎯 Dropoff Location:",
                list(landmarks.keys()),
                index=1
            )
            dropoff_coords = landmarks[dropoff_choice]
            
            # Display coordinates
            st.markdown("**Pickup Coordinates:**")
            st.write(f"Lat: {pickup_coords[0]:.4f}, Lon: {pickup_coords[1]:.4f}")
            
            st.markdown("**Dropoff Coordinates:**")
            st.write(f"Lat: {dropoff_coords[0]:.4f}, Lon: {dropoff_coords[1]:.4f}")
            
        else:
            # Custom coordinates
            st.markdown("**Pickup Location:**")
            pickup_lat = st.number_input("Latitude", value=1.2966, min_value=1.15, max_value=1.47, step=0.0001, key="pickup_lat")
            pickup_lon = st.number_input("Longitude", value=103.7764, min_value=103.6, max_value=104.1, step=0.0001, key="pickup_lon")
            pickup_coords = (pickup_lat, pickup_lon)
            
            st.markdown("**Dropoff Location:**")
            dropoff_lat = st.number_input("Latitude", value=1.2494, min_value=1.15, max_value=1.47, step=0.0001, key="dropoff_lat")
            dropoff_lon = st.number_input("Longitude", value=103.8303, min_value=103.6, max_value=104.1, step=0.0001, key="dropoff_lon")
            dropoff_coords = (dropoff_lat, dropoff_lon)
        
        # Trip details
        st.markdown("## ⏰ Trip Details")
        
        # Date and time
        col1, col2 = st.columns(2)
        with col1:
            trip_date = st.date_input("Pickup Date", value=datetime.now().date())
        with col2:
            trip_time = st.time_input("Pickup Time", value=datetime.now().time())
        
        # Combine date and time
        trip_datetime = datetime.combine(trip_date, trip_time)
        
        # Number of passengers
        passengers = st.number_input("Number of Passengers", min_value=1, max_value=4, value=1)
        
        # Calculate button
        calculate_button = st.button("🚀 Calculate Fare", type="primary")
    
    # Main content area
    if calculate_button:
        # Calculate distance
        distance = calculate_distance(pickup_coords[0], pickup_coords[1], 
                                    dropoff_coords[0], dropoff_coords[1])
        
        # Estimate duration
        duration = estimate_duration(distance)
        
        # Get time information
        hour = trip_datetime.hour
        day_of_week = trip_datetime.weekday()
        month = trip_datetime.month
        is_peak = is_peak_hour(hour)
        is_weekend_day = is_weekend(day_of_week)
        
        # Display trip summary
        st.markdown("## 📊 Trip Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Distance", f"{distance:.2f} km")
        with col2:
            st.metric("Duration", f"{duration:.1f} min")
        with col3:
            st.metric("Passengers", passengers)
        with col4:
            time_status = "Peak Hours" if is_peak else "Normal Hours"
            st.metric("Time Status", time_status)
        
        # Fare predictions
        st.markdown("## 💰 Fare Predictions")
        
        if ml_available:
            # ML predictions
            try:
                ml_predictions = ml_calculator.predict_fare_ml(
                    distance_km=distance,
                    duration_minutes=duration,
                    passengers=passengers,
                    hour=hour,
                    day_of_week=day_of_week,
                    month=month,
                    is_peak_hour=is_peak,
                    is_weekend=is_weekend_day
                )
                
                # Display ML predictions
                st.markdown("### 🤖 Machine Learning Models")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="ml-prediction-box">
                        <h3>Random Forest</h3>
                        <h2>${ml_predictions['Random Forest']} SGD</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="ml-prediction-box">
                        <h3>XGBoost</h3>
                        <h2>${ml_predictions['XGBoost']} SGD</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="ml-prediction-box">
                        <h3>Average</h3>
                        <h2>${ml_predictions['Average']} SGD</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Model comparison chart
                st.markdown("### 📈 Model Comparison")
                
                models = list(ml_predictions.keys())
                values = list(ml_predictions.values())
                
                fig = px.bar(
                    x=models, 
                    y=values,
                    title="Fare Predictions by Model",
                    labels={'x': 'Model', 'y': 'Fare (SGD)'},
                    color=values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"ML prediction failed: {e}")
                ml_available = False
        
        if not ml_available:
            # Fallback to simple formula
            st.markdown("### 📊 Simple Formula (Fallback)")
            
            simple_fare = calculate_fare(distance, duration, is_peak, is_weekend_day)
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Estimated Fare</h3>
                <h2>${simple_fare} SGD</h2>
                <p>Using simple formula calculation</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional information
        st.markdown("## ℹ️ Additional Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("### 🕐 Time Details")
            st.write(f"**Date:** {trip_date.strftime('%A, %B %d, %Y')}")
            st.write(f"**Time:** {trip_time.strftime('%I:%M %p')}")
            st.write(f"**Peak Hours:** {'Yes' if is_peak else 'No'}")
            st.write(f"**Weekend:** {'Yes' if is_weekend_day else 'No'}")
        
        with info_col2:
            st.markdown("### 📍 Location Details")
            if input_method == "Popular Landmarks":
                st.write(f"**From:** {pickup_choice}")
                st.write(f"**To:** {dropoff_choice}")
            else:
                st.write(f"**Pickup:** ({pickup_coords[0]:.4f}, {pickup_coords[1]:.4f})")
                st.write(f"**Dropoff:** ({dropoff_coords[0]:.4f}, {dropoff_coords[1]:.4f})")
            st.write(f"**Distance:** {distance:.2f} km")
            st.write(f"**Duration:** {duration:.1f} minutes")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🚕 Singapore Taxi Fare Predictor - Machine Learning Edition</p>
        <p>Built with Random Forest and XGBoost models</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
