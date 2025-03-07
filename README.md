Real-Time Air Quality Dashboard and Prediction System
This project is a real-time air quality monitoring and prediction system. It fetches live air quality data from an external API, processes the data, stores it in MongoDB, and displays a dashboard via a Flask web application. In addition, the system uses two machine learning models—BiLSTM and XGBoost—to predict future air quality index (AQI) values. The predictions from both models can be combined to generate a more robust forecast.

Project Overview
Real-Time Monitoring: Continuously fetches air quality data (e.g., PM2.5, PM10, NO2, Ozone, SO2) and calculates the overall AQI.
Data Storage: Saves the fetched data along with calculated AQI values to a MongoDB database.
Dashboard: A simple web interface (using Flask and HTML templates) displays current air quality metrics.
Prediction Models:
BiLSTM Model: A deep learning model for capturing temporal patterns in air quality data.
XGBoost Model: A gradient boosting model used to enhance prediction performance.
Combined Prediction: The predictions from both models can be averaged (or weighted) to produce a final forecast.
Directory Structure
pgsql
Copy
Edit
aq/
├── api.py                      # Main Flask application handling routes and prediction API.
├── bilstm_model.keras          # Trained BiLSTM model file.
├── Implementation.ipynb        # Jupyter Notebook for model training, data exploration, or analysis.
├── mongo.py                    # Module for MongoDB interactions (e.g., connecting and saving data).
├── new.ipynb                   # Additional Jupyter Notebook (possibly for experiments or further analysis).
├── processed_data.csv          # Processed dataset used for training the models.
├── scaler.pkl                  # Saved MinMaxScaler for feature normalization.
├── xgb_model.json              # Trained XGBoost model file.
│
├── templates/                  # Folder for HTML templates.
│   └── index.html              # Main dashboard HTML template.
│
└── untitled_project/           # Folder containing experiment and tuning trial configurations.
    ├── oracle.json             # Configuration file for model oracle settings.
    ├── tuner0.json             # Tuner configuration file.
    ├── trial_0/                # Contains configuration and checkpoint for trial 0.
    │   ├── build_config.json
    │   ├── checkpoint.weights.h5
    │   └── trial.json
    ├── trial_1/                # Contains configuration and checkpoint for trial 1.
    │   ├── build_config.json
    │   ├── checkpoint.weights.h5
    │   └── trial.json
    ├── trial_2/                # Contains configuration and checkpoint for trial 2.
    │   ├── build_config.json
    │   ├── checkpoint.weights.h5
    │   └── trial.json
    ├── trial_3/                # Contains configuration and checkpoint for trial 3.
    │   ├── build_config.json
    │   ├── checkpoint.weights.h5
    │   └── trial.json
    └── trial_4/                # Contains configuration and checkpoint for trial 4.
        ├── build_config.json
        ├── checkpoint.weights.h5
        └── trial.json

Install Dependencies
flask
pymongo
numpy
requests
tensorflow
keras
joblib
xgboost
scikit-learn

Set Up MongoDB
Ensure MongoDB is installed and running locally or update the connection string in api.py/mongo.py accordingly.

Obtain an API Key
Sign up at OpenWeatherMap (or your chosen data provider) and replace the API key in your code.

Usage
Run the Flask Application
Start the real-time dashboard and prediction API:
python api.py
The application will start on http://0.0.0.0:5000 (or http://localhost:5000). Visit this URL in your web browser to see the dashboard.

Endpoints
/ : Renders the main dashboard.
/sensor_data : Returns the latest air quality data from MongoDB in JSON format.
/predict : Expects a POST request with air quality feature data (e.g., PM2.5, PM10, NO2, SO2, Ozone) to return a combined AQI prediction.
Prediction
The prediction endpoint uses both the BiLSTM and XGBoost models to generate forecasts. Their predictions can be combined (e.g., averaged or weighted) to produce a final AQI prediction.

Model Training and Experiments
Jupyter Notebooks:
Implementation.ipynb and new.ipynb contain code for data analysis, feature engineering, and model training.
Untitled Project Folder:
This directory holds multiple trial configurations and checkpoints for tuning experiments. Each trial folder (e.g., trial_0, trial_1, etc.) contains configuration files (build_config.json, trial.json) and model checkpoints (checkpoint.weights.h5) used during hyperparameter tuning.

Future Enhancements
Improved Model Fusion: Experiment with different strategies to combine predictions from the BiLSTM and XGBoost models.
Dashboard Enhancements: Add more visualizations and interactive elements.
Deployment: Containerize the application with Docker and deploy to a cloud platform.
