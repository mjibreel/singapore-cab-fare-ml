"""
Singapore Taxi Fare Prediction - Streamlit Web Application
Beautiful web interface for taxi fare predictions in Singapore.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our utility functions
from haversine import calculate_distance, get_singapore_landmarks
from fare_calculator import calculate_fare, estimate_duration, is_peak_hour, is_weekend

# Page configuration
st.set_page_config(
    page_title="Singapore Taxi Fare Predictor",
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
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stSelectbox > div > div > select {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🚕 Singapore Taxi Fare Predictor</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Predict taxi fares in Singapore using two different pricing models. 
            Get accurate estimates for your trips with real-time calculations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for input
    with st.sidebar:
        st.header("📍 Trip Details")
        
        # Location selection method
        location_method = st.radio(
            "Choose location input method:",
            ["Popular Landmarks", "Custom Coordinates"]
        )
        
        if location_method == "Popular Landmarks":
            landmarks = get_singapore_landmarks()
            
            pickup_landmark = st.selectbox(
                "Pickup Location:",
                list(landmarks.keys())
            )
            dropoff_landmark = st.selectbox(
                "Dropoff Location:",
                list(landmarks.keys())
            )
            
            pickup_lat, pickup_lon = landmarks[pickup_landmark]
            dropoff_lat, dropoff_lon = landmarks[dropoff_landmark]
            
            st.info(f"**Pickup:** {pickup_landmark}")
            st.info(f"**Dropoff:** {dropoff_landmark}")
            
        else:
            st.markdown("**Enter custom coordinates:**")
            st.markdown("Singapore coordinates range:")
            st.markdown("- Latitude: 1.15°N to 1.47°N")
            st.markdown("- Longitude: 103.6°E to 104.1°E")
            
            col1, col2 = st.columns(2)
            with col1:
                pickup_lat = st.number_input("Pickup Latitude:", 1.15, 1.47, 1.2838, 0.001)
                dropoff_lat = st.number_input("Dropoff Latitude:", 1.15, 1.47, 1.4043, 0.001)
            with col2:
                pickup_lon = st.number_input("Pickup Longitude:", 103.6, 104.1, 103.8591, 0.001)
                dropoff_lon = st.number_input("Dropoff Longitude:", 103.6, 104.1, 103.7930, 0.001)
        
        # Trip details
        st.markdown("**Trip Details:**")
        pickup_date = st.date_input("Pickup Date:", datetime.now())
        pickup_time = st.time_input("Pickup Time:", datetime.now().replace(hour=12, minute=0))
        passenger_count = st.slider("Number of Passengers:", 1, 4, 1)
        
        # Peak hours information
        st.markdown("**Peak Hours Information:**")
        st.markdown("🕐 **Peak Hours** (higher fares):")
        st.markdown("- Morning: 7:00 AM - 9:00 AM")
        st.markdown("- Evening: 6:00 PM - 8:00 PM")
        st.markdown("- Model 1: +20% surcharge")
        st.markdown("- Model 2: +25% surcharge")
        
        # Calculate button
        calculate_button = st.button("🚀 Calculate Fare", type="primary")
    
    # Main content area
    if calculate_button:
        # Combine date and time
        pickup_datetime = datetime.combine(pickup_date, pickup_time)
        
        # Calculate distance
        distance = calculate_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        
        # Estimate duration
        duration = estimate_duration(distance)
        
        # Check if peak hour or weekend
        hour = pickup_datetime.hour
        day_of_week = pickup_datetime.weekday()
        peak = is_peak_hour(hour)
        weekend = is_weekend(day_of_week)
        
        # Calculate fare using TWO MODELS
        # Model 1: Basic formula
        basic_fare = calculate_fare(distance, duration, peak, weekend)
        
        # Model 2: Alternative pricing model
        alt_base_fare = 4.00
        alt_distance_cost = 1.80 * distance
        alt_time_cost = 0.35 * duration
        alt_fare = alt_base_fare + alt_distance_cost + alt_time_cost
        
        if peak:
            alt_fare *= 1.25
        if weekend:
            alt_fare *= 1.15
        
        alt_fare = round(alt_fare, 2)
        
        # Display results
        st.markdown("## 🚕 Trip Summary")
        
        # Trip information in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Distance", f"{distance:.2f} km")
        
        with col2:
            st.metric("Duration", f"{duration:.1f} min")
        
        with col3:
            st.metric("Passengers", passenger_count)
        
        with col4:
            peak_status = "Peak Hours" if peak else "Normal Hours"
            st.metric("Time Status", peak_status)
        
        # Fare predictions
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("## 💰 Fare Predictions (Two Models)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model 1 (Standard)", f"${basic_fare:.2f} SGD")
        
        with col2:
            st.metric("Model 2 (Premium)", f"${alt_fare:.2f} SGD")
        
        with col3:
            avg_fare = (basic_fare + alt_fare) / 2
            st.metric("Average Prediction", f"${avg_fare:.2f} SGD")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Trip details
        st.markdown("## 📍 Trip Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Location Information:**")
            if location_method == "Popular Landmarks":
                st.write(f"**Pickup:** {pickup_landmark}")
                st.write(f"**Dropoff:** {dropoff_landmark}")
            else:
                st.write(f"**Pickup:** ({pickup_lat:.4f}, {pickup_lon:.4f})")
                st.write(f"**Dropoff:** ({dropoff_lat:.4f}, {dropoff_lon:.4f})")
            
            st.write(f"**Date:** {pickup_date.strftime('%B %d, %Y')}")
            st.write(f"**Time:** {pickup_time.strftime('%I:%M %p')}")
            st.write(f"**Day:** {pickup_datetime.strftime('%A')}")
        
        with col2:
            st.markdown("**Trip Information:**")
            st.write(f"**Distance:** {distance:.2f} km")
            st.write(f"**Duration:** {duration:.1f} minutes")
            st.write(f"**Peak Hours:** {'Yes' if peak else 'No'}")
            st.write(f"**Weekend:** {'Yes' if weekend else 'No'}")
        
        # Additional information
        st.markdown("## ℹ️ Additional Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Information:**")
            st.write("- **Model 1 (Standard)**: Regular taxi pricing with 20% peak surcharge")
            st.write("- **Model 2 (Premium)**: Premium service pricing with 25% peak surcharge")
            st.write("- **Success Target**: RMSE < $3.00 SGD")
            st.write("- **Data Source**: Singapore Taxi Trip Records + Synthetic Fare Generation")
        
        with col2:
            st.markdown("**Technical Features:**")
            st.write("- **Distance Calculation**: Haversine formula for accurate Earth distances")
            st.write("- **Peak Hour Detection**: Automatic 7-9 AM and 6-8 PM detection")
            st.write("- **Weekend Detection**: Saturday/Sunday surcharge application")
            st.write("- **Real-time Calculation**: Instant updates based on user inputs")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Singapore Taxi Fare Predictor</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
