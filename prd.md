Here is a detailed Product Requirements Document (PRD) for your Singapore Taxi Fare Prediction project. This document will serve as your project's blueprint, defining exactly what you need to build, why, and for whom.

Product Requirements Document (PRD)
Project Title: Machine Learning Pipeline for Singapore Taxi Fare Prediction
Version: 1.0
Date: May 23, 2024

1. Problem Statement & Objective
Problem: Tourists and residents in Singapore lack a reliable way to estimate taxi fares before booking a ride. This leads to budget uncertainty and potential dissatisfaction with ride-hailing services.

Objective: To develop a accurate and user-friendly machine learning model that predicts taxi fares in Singapore based on trip details. The project will demonstrate a complete ML pipeline from data acquisition to a functional web application.

Success Criteria: A deployed web application where users can input trip details and receive a fare prediction with an Average Prediction Error of less than ±$3 SGD.

2. User Persona & Use Case
User: A tourist in Singapore planning a trip from Marina Bay Sands to the Singapore Zoo.

Use Case: The user opens our web app, enters the pickup and dropoff locations (or coordinates), the number of passengers, and the date/time. The application returns a predicted fare amount, helping the user budget for their trip.

3. Scope & Deliverables (Aligned with Assignment Brief)
Deliverable	Description	Format
1. Proposal Report	A short document outlining the project plan.	PDF Document
2. GitHub Repository	A well-organized repo containing all code, models, and documentation.	Public GitHub Link
3. Trained Model Files	Serialized files of the trained machine learning models.	.pkl or .joblib files
4. Streamlit Web App	A functional web application for making predictions.	Public URL (e.g., *.streamlit.app)
5. Presentation Slides	A summary of the project for the final presentation.	PowerPoint or PDF
4. Technical Requirements & Methodology
A. Data Source:

Primary Dataset: A sample from the Singapore Taxi Trip Records dataset on Kaggle.

Synthetic Data Generation: Due to the lack of a real fare_amount column, we will synthetically generate the target variable using a plausible pricing model:

fare_amount = 3.20 + (1.50 * trip_distance_km) + (0.40 * trip_duration_minutes)

A random surge multiplier (1.0x - 2.0x) will be applied to 10% of trips to simulate peak pricing.

B. Machine Learning Models (Two Required):

Model 1: Random Forest Regressor

Justification: Powerful and versatile; handles non-linear relationships well and is robust to outliers. A strong baseline model for tabular data.

Model 2: Gradient Boosting Regressor (XGBoost or LightGBM)

Justification: Often provides state-of-the-art performance on regression tasks by sequentially correcting the errors of previous models. We will compare its performance to the Random Forest model.

C. Evaluation Metrics:

Primary Metric: Root Mean Squared Error (RMSE) - punishes large errors.

Secondary Metrics: Mean Absolute Error (MAE) - easy to interpret, R-Squared (R²) - explains variance.

5. Project Plan & Timeline (High-Level)
Week 1: Data & Setup

Form team, set up GitHub repo with required folder structure.

Acquire dataset, perform exploratory data analysis (EDA), and generate synthetic fare_amount.

Deliverable: Proposal Report

Week 2: Modeling

Complete data preprocessing and feature engineering (distance calculation, time features).

Train, validate, and compare the two models (Random Forest & Gradient Boosting).

Select the best model and save it.

Week 3: Deployment & Documentation

Develop the Streamlit application.

Write a comprehensive README.md.

Finalize code and documentation in GitHub.

Deliverable: GitHub Repository & Streamlit App

Week 4: Presentation

Prepare and practice the final presentation.

Deliverable: Presentation Slides

6. GitHub Repository Structure (Required)
text
singapore-taxi-fare-prediction/
├── README.md                       # Project documentation & report
├── app.py                          # Streamlit application
├── /data                           # Folder for dataset (optional, use .gitignore)
├── /notebooks
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Model_Testing.ipynb
├── /models
│   ├── random_forest_model.pkl
│   └── gradient_boosting_model.pkl
├── /src                            # For utility scripts (e.g., haversine.py)
└── /slides
    └── Final_Presentation.pdf
