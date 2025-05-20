# Real-Time Air Quality Dashboard and Prediction System 🚀

This project is a **real-time air quality monitoring and prediction system**. It fetches live air quality data from an external API, processes and stores the data in **MongoDB**, and provides a **Flask-based dashboard** for visualization. The system employs hybrid approach by combining time-based deep learning models—**BiLSTM** and machine learning **XGBoost**— model on past 5 years real dataset taken from CPCB govt. website for Delhi region, Trained to predict future AQI (Air Quality Index) values, improving forecast accuracy by combining their outputs.

---

## 🌍 Project Overview

### ✅ **Key Features**
- **Real-Time Monitoring**: Fetches live air quality data (e.g., PM2.5, PM10, NO2, Ozone, SO2) from open weather using Api key and calculates AQI.
- **Data Storage**: Stores processed data in a **MongoDB database**.
- **Interactive Dashboard**: A Flask web app provides real-time AQI visualizations.
- **Machine Learning Predictions**:
  - **BiLSTM** (Deep Learning) for capturing temporal patterns.
  - **XGBoost** (Gradient Boosting) for enhanced accuracy.
  - **Combined Prediction** for improved forecasting.

---

## 📂 Directory Structure

```
├── api.py                      # Flask application (API & dashboard)
├── bilstm_model.keras          # Trained BiLSTM model file
├── Implementation.ipynb        # Jupyter Notebook for model training & analysis
├── mongo.py                    # MongoDB interaction module
├── new.ipynb                   # Additional Jupyter Notebook for experiments
├── processed_data.csv          # Processed dataset for training
├── scaler.pkl                  # MinMaxScaler for feature normalization
├── xgb_model.json              # Trained XGBoost model file
│
├── templates/                  # HTML templates for dashboard
│   └── index.html              # Main dashboard UI
│
└── untitled_project/           # Hyperparameter tuning trials
    ├── oracle.json             # Model tuning oracle settings
    ├── tuner0.json             # Tuner configurations
    ├── trial_X/                # Checkpoints and settings for multiple trials
        ├── build_config.json
        ├── checkpoint.weights.h5
        └── trial.json
```

### **2️⃣ Set Up MongoDB**
- Ensure MongoDB is **installed and running**.
- Update the MongoDB connection string in `api.py` and `mongo.py` if needed.

### **3️⃣ Obtain an API Key**
Sign up at [OpenWeatherMap](https://openweathermap.org/) (or any air quality data provider) and replace the API key in your code.

---

## 🚀 Usage

### **Run the Flask Application**
Start the real-time dashboard and prediction API:
```bash
python api.py
```
The application will be available at: [http://localhost:5000](http://localhost:5000)

### **Available Endpoints**
| Endpoint         | Description |
|-----------------|-------------|
| `/`             | Renders the dashboard |
| `/sensor_data`  | Returns latest air quality data from MongoDB (JSON) |
| `/predict`      | Accepts air quality features (PM2.5, PM10, NO2, SO2, Ozone) via **POST** and returns AQI prediction |

---

## 📊 Prediction System
- The **prediction endpoint** utilizes both **BiLSTM** and **XGBoost** models to forecast AQI.
- Their outputs can be **averaged or weighted** to generate an optimal prediction.

---

## 🧠 Model Training & Experiments

### **Jupyter Notebooks**
- `Implementation.ipynb` and `new.ipynb` contain **data preprocessing, feature engineering, and model training** steps.

### **Hyperparameter Tuning Trials**
- `untitled_project/` contains **multiple tuning trials** with:
  - **Configuration files** (e.g., `build_config.json`, `trial.json`)
  - **Model checkpoints** (`checkpoint.weights.h5` for each trial)

---

## 🔮 Future Enhancements
- **Advanced Model Fusion**: Experiment with ensemble techniques for better AQI prediction.
- **Enhanced Dashboard**: Add interactive graphs and time-series forecasting.
- **Deployment**: Containerize with **Docker** and deploy on cloud platforms like AWS/GCP.


