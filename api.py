from flask import Flask, jsonify, redirect, url_for, render_template, request
from pymongo import MongoClient
import numpy as np
import pandas as pd
import requests
import time
import threading
from datetime import datetime
from bson import ObjectId
from tensorflow.keras.models import load_model
import joblib
import xgboost as xgb
app = Flask(__name__)

client = MongoClient('localhost', 27017)
db = client['air_quality']
collection = db['delhi_data']

api_key = "30a138a76e35b2db42e4c713c73fb3f9"
latitude = 28.6139
longitude = 77.2090

scaler = joblib.load('C:\\Users\\vidhi\\Downloads\\vscode\\aq\\scaler.pkl')

# Load the trained model (make sure to replace 'model.pkl' with your actual model file)
bilstm_model = load_model('C:\\Users\\vidhi\\Downloads\\vscode\\aq\\bilstm_model.keras')

# Load the XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('C:\\Users\\vidhi\\Downloads\\vscode\\aq\\xgb_model.json')



# Define breakpoints for PM2.5, PM10, NO2, O3, SO2 based on Indian AQI standards
breakpoints = {
    "pm25": [
        {"BLO": 0, "BHI": 30, "ILO": 0, "IHI": 50},
        {"BLO": 31, "BHI": 60, "ILO": 51, "IHI": 100},
        {"BLO": 61, "BHI": 90, "ILO": 101, "IHI": 200},
        {"BLO": 91, "BHI": 120, "ILO": 201, "IHI": 300},
        {"BLO": 121, "BHI": 250, "ILO": 301, "IHI": 400},
        {"BLO": 251, "BHI": 500, "ILO": 401, "IHI": 500},
    ],
    "pm10": [
        {"BLO": 0, "BHI": 50, "ILO": 0, "IHI": 50},
        {"BLO": 51, "BHI": 100, "ILO": 51, "IHI": 100},
        {"BLO": 101, "BHI": 250, "ILO": 101, "IHI": 200},
        {"BLO": 251, "BHI": 350, "ILO": 201, "IHI": 300},
        {"BLO": 351, "BHI": 430, "ILO": 301, "IHI": 400},
        {"BLO": 431, "BHI": 500, "ILO": 401, "IHI": 500},
    ],
    "no2": [
        {"BLO": 0, "BHI": 40, "ILO": 0, "IHI": 50},
        {"BLO": 41, "BHI": 80, "ILO": 51, "IHI": 100},
        {"BLO": 81, "BHI": 180, "ILO": 101, "IHI": 200},
        {"BLO": 181, "BHI": 280, "ILO": 201, "IHI": 300},
        {"BLO": 281, "BHI": 400, "ILO": 301, "IHI": 400},
    ],
    "o3": [
        {"BLO": 0, "BHI": 50, "ILO": 0, "IHI": 50},
        {"BLO": 51, "BHI": 100, "ILO": 51, "IHI": 100},
        {"BLO": 101, "BHI": 168, "ILO": 101, "IHI": 200},
        {"BLO": 169, "BHI": 208, "ILO": 201, "IHI": 300},
        {"BLO": 209, "BHI": 748, "ILO": 301, "IHI": 400},
    ],
    "so2": [
        {"BLO": 0, "BHI": 40, "ILO": 0, "IHI": 50},
        {"BLO": 41, "BHI": 80, "ILO": 51, "IHI": 100},
        {"BLO": 81, "BHI": 380, "ILO": 101, "IHI": 200},
        {"BLO": 381, "BHI": 800, "ILO": 201, "IHI": 300},
        {"BLO": 801, "BHI": 1600, "ILO": 301, "IHI": 400},
    ]
}

# Function to calculate AQI based on concentration and pollutant breakpoints
def calculate_aqi(concentration, pollutant):
    for bp in breakpoints[pollutant]:
        if bp["BLO"] <= concentration <= bp["BHI"]:
            return ((bp["IHI"] - bp["ILO"]) / (bp["BHI"] - bp["BLO"])) * (concentration - bp["BLO"]) + bp["ILO"]
    return None  # If out of range

def get_current_air_quality(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# Use the provided MongoDB saving function
def save_to_mongo(pm25, pm10, no2, o3, so2, aqi_pm25, aqi_pm10, aqi_no2, aqi_o3, aqi_so2, overall_aqi):
    try:
        # Prepare data document to be saved in MongoDB
        air_quality_data = {
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "o3": o3,
            "so2": so2,
            "aqi_pm25": aqi_pm25,
            "aqi_pm10": aqi_pm10,
            "aqi_no2": aqi_no2,
            "aqi_o3": aqi_o3,
            "aqi_so2": aqi_so2,
            "overall_aqi": overall_aqi,
            "timestamp": datetime.now()  # Add a timestamp field
        }

        # Insert the document into the MongoDB collection
        collection.insert_one(air_quality_data)
        print("Air quality data saved to MongoDB.")
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")

def fetch_and_save_air_quality():
    while True:
        current_data = get_current_air_quality(latitude, longitude, api_key)
        if current_data:
            # Fetch pollutant concentrations
            pm25 = current_data['list'][0]['components']['pm2_5']
            pm10 = current_data['list'][0]['components']['pm10']
            no2 = current_data['list'][0]['components']['no2']
            o3 = current_data['list'][0]['components']['o3']
            so2 = current_data['list'][0]['components']['so2']

            # Calculate AQI for each pollutant
            aqi_pm25 = calculate_aqi(pm25, 'pm25')
            aqi_pm10 = calculate_aqi(pm10, 'pm10')
            aqi_no2 = calculate_aqi(no2, 'no2')
            aqi_o3 = calculate_aqi(o3, 'o3')
            aqi_so2 = calculate_aqi(so2, 'so2')

            # The overall AQI is the maximum of individual pollutant AQIs
            overall_aqi = max(filter(None, [aqi_pm25, aqi_pm10, aqi_no2, aqi_o3, aqi_so2]))

            # Save to MongoDB
            save_to_mongo(pm25, pm10, no2, o3, so2, aqi_pm25, aqi_pm10, aqi_no2, aqi_o3, aqi_so2, overall_aqi)

        time.sleep(3600)

# Custom JSON serializer for ObjectId
def json_serializer(data):
    if isinstance(data, ObjectId):
        return str(data)
    raise TypeError(f"Type {type(data)} not serializable")

# Serve the dashboard HTML
@app.route('/')
def index():
    return render_template('index.html')

def get_last_24_data():
    # Fetch the last 24 documents from the MongoDB collection
    last_24_entries = collection.find().sort('timestamp', -1).limit(24)
    
    # Prepare data in the format (24, 8)
    data = []
    for entry in last_24_entries:
        data.append([
            entry.get('pm25', 0),    # PM2.5, default to 0 if missing
            entry.get('pm10', 0),    # PM10, default to 0 if missing
            entry.get('no2', 0),     # NO2, default to 0 if missing
            entry.get('o3', 0),      # Ozone, default to 0 if missing
            entry.get('so2', 0),     # SO2, default to 0 if missing
            entry.get('aqi_pm25', 0), # AQI for PM2.5, default to 0 if missing
            entry.get('aqi_pm10', 0), # AQI for PM10, default to 0 if missing
            entry.get('overall_aqi', 0) # Overall AQI, default to 0 if missing
        ])

    # Convert to a numpy array and reshape to (24, 8)
    import numpy as np
    data_array = np.array(data)
    
    # Check if we have less than 24 entries and pad if needed
    if data_array.shape[0] < 24:
        data_array = np.pad(data_array, ((0, 24 - data_array.shape[0]), (0, 0)), 'constant')

    return data_array[-24:]  # Return the last 24 entries

# Endpoint to get the latest air quality data
@app.route('/sensor_data', methods=['GET'])
def get_latest_air_quality():
    latest_data = collection.find().sort([('timestamp', -1)]).limit(1)
    data = list(latest_data)
    
    if data:
        data[0]['_id'] = str(data[0]['_id'])
        return jsonify(data[0])
    else:
        return jsonify({'error': 'No data available'}), 404


@app.route('/predict', methods=['POST'])
def predict_aqi():
    try:
        # Extract input values from JSON request
        data = request.get_json()
        required_keys = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'overall_aqi']

        # Validate input data
        for key in required_keys:
            if key not in data:
                return jsonify({'error': f'Missing value for {key}'}), 400

        # Parse input data
        pm25 = float(data['pm25'])
        pm10 = float(data['pm10'])
        no2 = float(data['no2'])
        o3 = float(data['o3'])
        so2 = float(data['so2'])
        overall_aqi = float(data['overall_aqi'])

        # Prepare input data for prediction
        features = np.array([[pm25, pm10, no2, o3, so2, overall_aqi]])  # Shape (1, 6)
        features_df = pd.DataFrame(features)

        # Apply scaling
        scaled_features = scaler.transform(features_df)

        # Reshape for BiLSTM model
        scaled_features = scaled_features.reshape((1, 1, 6))  # Reshape to (batch_size, timesteps, features)

        # Predict with BiLSTM
        bilstm_predicted_aqi = bilstm_model.predict(scaled_features)

        # Reshape output for XGBoost
        bilstm_predicted_aqi = bilstm_predicted_aqi.reshape(bilstm_predicted_aqi.shape[0], -1)

        # Predict with XGBoost
        xgboost_predicted_aqi = xgb_model.predict(bilstm_predicted_aqi)

        # Prepare features for inverse scaling
        scaled_features_for_inverse = np.zeros((1, len(features[0])))
        scaled_features_for_inverse[0, -1] = xgboost_predicted_aqi  # Set AQI for inverse scaling

        # Inverse scaling
        y_pred_rescaled = scaler.inverse_transform(scaled_features_for_inverse)[:, -1]

        # Return the predicted AQI
        return jsonify({'predicted_aqi': round(y_pred_rescaled[0], 2)})

    except Exception as e:
        print(f"Error: {str(e)}")  # Log error for debugging
        return jsonify({'error': str(e)}), 500

# Start air quality data fetching thread
def start_fetching_thread():
    fetching_thread = threading.Thread(target=fetch_and_save_air_quality)
    fetching_thread.daemon = True
    fetching_thread.start()

if __name__ == '__main__':
    start_fetching_thread()
    app.run(debug=True, host='0.0.0.0')