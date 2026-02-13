🎮 Intelligent Gaming Performance Optimizer
AI-Powered System for Automatic Game Settings Optimization
📌 Project Overview

Modern PC games provide numerous graphical settings (resolution, texture quality, shadows, render scale, etc.). However, selecting the optimal configuration for a specific hardware setup is complex and often requires trial and error.

This project builds a Machine Learning-based performance optimizer that:

Monitors real-time system metrics (CPU, RAM, GPU, FPS)

Collects gameplay performance data

Trains ML models to learn performance patterns

Recommends optimal game settings

Balances FPS, system temperature, and visual quality

The goal is to provide intelligent, data-driven gaming optimization instead of manual experimentation.

🧠 Machine Learning Logic Used
🎯 Problem Type:

Supervised Learning – Regression

We predict:

Average FPS

Performance Score

📊 Features Used:

Resolution (e.g., 1280x720)

Graphics Quality (Low/Medium/High)

Render Scale

CPU Usage %

RAM Usage %

GPU Usage %

Temperature

Other performance metrics

🤖 Model Used:

Random Forest Regressor

XGBoost (optional advanced version)

The model learns how system metrics + settings impact FPS and performance stability.

🏗️ System Architecture

Collect performance logs during gameplay

Store runs with RUN_ID

Preprocess dataset

Encode categorical variables

Train ML model

Save model using Joblib

Deploy with Streamlit UI

Provide recommended optimal settings

⚙️ Technologies Used
💻 Programming Language

Python 3.x

📦 Core Libraries

pandas (data processing)

numpy (numerical operations)

scikit-learn (ML models)

xgboost (advanced boosting model)

joblib (model persistence)

psutil (system monitoring)

plotly (interactive graphs)

streamlit (web app UI)

🧠 ML Algorithms

Random Forest Regression

Gradient Boosting (XGBoost)

📊 Data Handling

CSV-based logging

Feature engineering

Label encoding / One-hot encoding

🚀 How to Run the Project
Step 1: Install Requirements
pip install -r requirements.txt

Step 2: Run Streamlit App
streamlit run app.streamlit.py

Step 3: Collect Performance Data

Start monitoring

Play game for fixed duration (e.g., 60 seconds)

Save run with unique RUN_ID

Step 4: Train Model
python train_model.py

📈 Output Features

FPS Prediction

System Resource Usage Analysis

Performance Graphs

Optimal Settings Recommendation

Performance Stability Score

🎯 Key Contributions

✔ Real-time performance monitoring
✔ Automated ML-based optimization
✔ User-friendly interactive UI
✔ Data-driven decision system
✔ Scalable model architecture

🔬 Future Enhancements

Reinforcement Learning for dynamic optimization

Automatic hardware detection

Cloud-based dataset aggregation

Cross-game learning system

Model comparison dashboard

📌 Academic Relevance

This project demonstrates:

Supervised Learning

Regression Modeling

Feature Engineering

Model Evaluation

Deployment of ML model using Streamlit

Real-time system data integration

It bridges practical gaming systems with applied Machine Learning.
