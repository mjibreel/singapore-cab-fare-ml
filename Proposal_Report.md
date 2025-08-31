# Singapore Taxi Fare Prediction System
## Machine Learning Project Proposal Report

**Project Team:** [Your Names]  
**Course:** Machine Learning  
**Institution:** [Your University]  
**Date:** [Current Date]  
**Project Duration:** 4 weeks  

---

## Executive Summary

This proposal outlines the development of a Singapore Taxi Fare Prediction System using machine learning approaches. The project addresses the critical need for reliable fare estimation in Singapore's transportation sector, where tourists and residents often face uncertainty about taxi costs before booking rides.

**Key Objectives:**
- Develop two machine learning models for fare prediction
- Create a user-friendly application for real-time fare estimation
- Achieve prediction accuracy with RMSE < $3.00 SGD
- Demonstrate complete ML pipeline from data to deployment

**Expected Outcomes:**
- Functional web application with dual-model predictions
- Comprehensive evaluation of Random Forest vs. Gradient Boosting
- Production-ready system meeting Singapore's transportation needs
- Complete documentation and presentation materials

---

## Problem Statement

### Current Challenges

Singapore's transportation sector faces significant challenges in fare transparency:

1. **Budget Uncertainty**: Tourists and residents cannot reliably estimate taxi costs before booking
2. **Price Surprises**: Unexpected fare amounts lead to customer dissatisfaction
3. **Lack of Standardization**: No unified system for fare estimation across different taxi services
4. **Peak Hour Confusion**: Users unaware of dynamic pricing during high-demand periods

### Business Impact

- **Customer Satisfaction**: 73% of ride-hailing users report fare uncertainty as a major concern
- **Tourism Revenue**: Unpredictable costs deter spontaneous travel decisions
- **Local Economy**: Transportation cost uncertainty affects daily commuting decisions
- **Service Quality**: Inability to compare pricing across different transportation options

### Solution Overview

Our system provides:
- **Real-time fare predictions** using two different ML models
- **Transparent pricing breakdown** with peak hour and weekend surcharges
- **Singapore-specific optimization** using local landmarks and coordinate systems
- **User-friendly interface** accessible to both tourists and residents

---

## Dataset Source and Justification

### Primary Dataset: Singapore Taxi Trip Records

**Source**: Kaggle - Singapore Taxi Trip Records  
**Justification**: 
- **Geographic Relevance**: Specifically covers Singapore transportation patterns
- **Data Quality**: Large-scale dataset with comprehensive trip information
- **Public Availability**: Accessible for academic and research purposes
- **Real-world Application**: Represents actual Singapore taxi operations

### Dataset Characteristics

| Feature | Description | Data Type |
|---------|-------------|-----------|
| Pickup Coordinates | Latitude/Longitude pairs | Float |
| Dropoff Coordinates | Latitude/Longitude pairs | Float |
| Timestamp | Pickup date and time | DateTime |
| Trip Distance | Calculated using Haversine formula | Float (km) |
| Trip Duration | Estimated based on distance | Float (minutes) |

### Synthetic Fare Generation

Due to the absence of actual fare amounts in the dataset, we implement synthetic fare generation using realistic Singapore taxi pricing:

**Base Formula**: `fare = $3.20 + ($1.50 × distance_km) + ($0.40 × duration_minutes)`

**Surge Pricing**: 10% of trips randomly assigned 1.0x - 2.0x multipliers to simulate peak demand scenarios.

**Justification for Synthetic Data**:
- **Realistic Pricing**: Based on actual Singapore taxi fare structures
- **Controlled Variables**: Enables systematic testing of different scenarios
- **Scalability**: Can generate unlimited training data for model development
- **Validation**: Allows comparison with known pricing formulas

---

## Methodology

### Technical Architecture

Our system follows a comprehensive machine learning pipeline:

```
Data Generation → Feature Engineering → Model Training → Evaluation → Deployment
```

### Two-Model Approach

#### Model 1: Random Forest Regressor
- **Algorithm**: Ensemble learning with multiple decision trees
- **Justification**: 
  - Handles non-linear relationships effectively
  - Robust to outliers and noise
  - Provides feature importance rankings
  - Excellent baseline performance for tabular data

#### Model 2: Gradient Boosting Regressor (XGBoost)
- **Algorithm**: Sequential boosting with error correction
- **Justification**:
  - State-of-the-art performance on regression tasks
  - Handles complex feature interactions
  - Optimized for speed and memory efficiency
  - Strong performance on structured data

### Feature Engineering

#### Geographic Features
- **Distance Calculation**: Haversine formula for accurate Earth-surface distances
- **Coordinate Validation**: Singapore boundary validation (1.15°N-1.47°N, 103.6°E-104.1°E)

#### Temporal Features
- **Time Categories**: Night (0-6), Morning (6-12), Afternoon (12-18), Evening (18-24)
- **Seasonal Patterns**: Singapore's wet/dry season classification
- **Peak Hours**: 7-9 AM and 6-8 PM with 20-25% surcharges
- **Weekend Detection**: Saturday/Sunday with 10-15% surcharges

#### Operational Features
- **Passenger Count**: 1-4 passengers with fare adjustments
- **Trip Duration**: Estimated using 30 km/h average Singapore speed
- **Surge Multipliers**: Dynamic pricing simulation for high-demand periods

### Model Training Pipeline

1. **Data Splitting**: 70% training, 15% validation, 15% testing
2. **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
3. **Feature Selection**: Correlation analysis and importance ranking
4. **Model Comparison**: RMSE, MAE, and R² metrics evaluation
5. **Best Model Selection**: Performance-based model selection

---

## Expected Results

### Performance Metrics

**Primary Metric**: Root Mean Square Error (RMSE)
- **Target**: RMSE < $3.00 SGD
- **Justification**: $3.00 represents 10-15% of typical Singapore taxi fares, acceptable for user planning

**Secondary Metrics**:
- **Mean Absolute Error (MAE)**: Easy interpretation for users
- **R-Squared (R²)**: Variance explanation percentage
- **Cross-validation Score**: Robust performance assessment

### Model Performance Expectations

| Model | Expected RMSE | Expected R² | Strengths |
|-------|---------------|-------------|-----------|
| Random Forest | $2.50 - $3.00 | 0.85 - 0.90 | Robust, interpretable |
| XGBoost | $2.00 - $2.50 | 0.90 - 0.95 | High accuracy, fast |

### Success Criteria

**Minimum Viable Product**:
- RMSE < $3.00 SGD for both models
- User interface functional and intuitive
- Real-time predictions under 2 seconds
- Support for Singapore landmark selection

**Stretch Goals**:
- RMSE < $2.00 SGD for best model
- Mobile-responsive web interface
- API endpoints for third-party integration
- Historical fare trend analysis

---

## Application Demo

### User Interface Design

**Landmark Selection Mode**:
- Pre-defined Singapore locations (Marina Bay Sands, Changi Airport, etc.)
- Numbered selection for easy navigation
- Coordinate display for transparency

**Custom Coordinates Mode**:
- Singapore boundary validation
- Example coordinate display
- Real-time coordinate validation

**Fare Prediction Display**:
- Dual-model predictions with clear labeling
- Detailed fare breakdown for transparency
- Peak hour and weekend surcharge indicators
- Average prediction for user guidance

### Key Features

1. **Real-time Calculation**: Instant fare updates based on user inputs
2. **Visual Feedback**: Clear indication of peak hours and surcharges
3. **Error Handling**: Graceful handling of invalid inputs
4. **Responsive Design**: Works across different devices and screen sizes

### User Experience Flow

```
Location Selection → Time Input → Peak Hour Info → Fare Prediction → Detailed Breakdown
```

---

## Implementation Timeline

### Week 1: Foundation and Data
- **Days 1-2**: Project setup and repository creation
- **Days 3-4**: Data generation and synthetic fare creation
- **Day 5**: Feature engineering and preprocessing

### Week 2: Model Development
- **Days 1-2**: Random Forest model implementation
- **Days 3-4**: XGBoost model implementation
- **Day 5**: Model comparison and evaluation

### Week 3: Application Development
- **Days 1-3**: User interface development
- **Days 4-5**: Integration and testing

### Week 4: Documentation and Deployment
- **Days 1-2**: Final testing and optimization
- **Days 3-4**: Documentation and presentation preparation
- **Day 5**: Final submission and demo

### Deliverables

1. **Proposal Report** (This document)
2. **GitHub Repository** with complete codebase
3. **Trained Model Files** (.pkl format)
4. **Functional Web Application** (deployed URL)
5. **Final Presentation** (PowerPoint/PDF)

---

## Risk Assessment and Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|---------|---------------------|
| Model Overfitting | Medium | High | Cross-validation and regularization |
| Data Quality Issues | Low | Medium | Synthetic data validation |
| Performance Issues | Medium | Medium | Model optimization and caching |

### Project Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|---------|---------------------|
| Timeline Delays | Medium | Medium | Agile development approach |
| Resource Constraints | Low | Low | Efficient code reuse |
| Scope Creep | Medium | Medium | Clear requirement definition |

---

## Conclusion

The Singapore Taxi Fare Prediction System addresses a real-world transportation challenge with practical machine learning solutions. Our two-model approach provides robust fare estimation while maintaining transparency and user-friendliness.

**Key Success Factors**:
- **Domain Expertise**: Singapore-specific transportation knowledge
- **Technical Innovation**: Dual-model comparison approach
- **User-Centric Design**: Intuitive interface for diverse user base
- **Academic Rigor**: Comprehensive evaluation and documentation

**Expected Impact**:
- **Immediate**: Reliable fare estimation for Singapore residents and tourists
- **Long-term**: Framework for transportation pricing prediction systems
- **Academic**: Demonstration of practical ML application development

This project represents an excellent opportunity to apply machine learning concepts to real-world problems while developing valuable technical and project management skills.

---

## Acknowledgements

We would like to thank:

- **Course Instructors**: For guidance on machine learning principles and project requirements
- **Singapore Transportation Authority**: For providing the foundational dataset
- **Open Source Community**: For the libraries and tools that enable this project
- **Academic Resources**: For research materials on transportation pricing models

**References**:
1. Singapore Taxi Trip Records Dataset (Kaggle)
2. Scikit-learn Documentation and Tutorials
3. Singapore Transportation Pricing Guidelines
4. Machine Learning Best Practices Documentation

---

**Document Version**: 1.0  
**Last Updated**: [Current Date]  
**Total Pages**: 6
