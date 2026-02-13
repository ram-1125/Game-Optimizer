🎮 Intelligent Gaming Performance Optimizer
AI-Powered System for Automatic Game Settings Optimization
📌 Project Overview

Modern PC games provide numerous graphical settings (resolution, texture quality, shadows, render scale, etc.). Selecting the optimal configuration for a specific hardware setup is complex and often requires trial and error.

This project builds a Machine Learning-based performance optimizer that:
Monitors real-time system metrics (CPU, RAM, GPU, FPS)
Collects gameplay performance data
Trains ML models to learn performance patterns
Recommends optimal game settings
Balances FPS, system temperature, and visual quality
The goal is to provide intelligent, data-driven gaming optimization instead of manual experimentation.

🧠 Machine Learning Logic Used
🎯 Problem Type
Supervised Learning – Regression
📈 Prediction Targets
Average FPS
Performance Score
📊 Features Used
Resolution (e.g., 1280x720)
Graphics Quality (Low / Medium / High)
Render Scale
CPU Usage (%)
RAM Usage (%)
GPU Usage (%)
Temperature
Other runtime performance metrics
🤖 Models Implemented
Random Forest Regressor
XGBoost Regressor 
The model learns how hardware metrics and game settings influence FPS and overall performance stability.

🏗️ System Architecture
Collect performance logs during gameplay
Store runs with unique RUN_ID
Preprocess dataset
Encode categorical variables
Train ML model
Save model using Joblib
Deploy using Streamlit UI
Provide recommended optimal settings

⚙️ Technologies Used
💻 Programming Language
Python 3.x
📦 Core Libraries
pandas – Data processing
numpy – Numerical computation
scikit-learn – Machine learning models
xgboost – Gradient boosting model
joblib – Model persistence
psutil – System monitoring
plotly – Interactive visualization
streamlit – Web application interface

🧠 ML Algorithms
Random Forest Regression
Gradient Boosting (XGBoost)

📊 Data Handling
CSV-based logging
Feature engineering
Label encoding / One-hot encoding

🚀 How to Run the Project
Step 1 – Install Requirements
pip install -r requirements.txt

Step 2 – Run Streamlit Application
streamlit run app.streamlit.py

Step 3 – Collect Performance Data

Start monitoring
Play the game for a fixed duration (e.g., 60 seconds)
Save run with a unique RUN_ID

Step 4 – Train the Model
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
✔ User-friendly interactive dashboard
✔ Data-driven decision system
✔ Scalable model architecture

🔬 Future Enhancements

Reinforcement Learning for dynamic optimization
Automatic hardware detection
Cloud-based dataset aggregation
Cross-game learning system
Model comparison dashboard
