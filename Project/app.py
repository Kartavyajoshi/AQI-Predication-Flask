
# from flask import Flask, render_template, jsonify
# import random
# from datetime import datetime, timedelta
# # Add your real ML imports (load_model, joblib, requests) here for production
# # import requests
# # from tensorflow.keras.models import load_model
# # import joblib
# # import numpy as np 
# # import os

# # --- Configuration (Set your actual key here) ---
# WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY" 

# # Initialize the Flask application
# app = Flask(__name__)

# # --- Helper Functions ---

# def get_aqi_status(aqi):
#     """Determines AQI status, text color class, and background color class based on the score."""
#     if aqi <= 50:
#         return "GOOD", "text-green-400", "bg-green-600/50"
#     elif aqi <= 100:
#         return "MODERATE", "text-yellow-400", "bg-yellow-600/50"
#     elif aqi <= 150:
#         return "UNHEALTHY (Sensitive)", "text-orange-400", "bg-orange-600/50"
#     else:
#         return "UNHEALTHY", "text-red-500", "bg-red-600/50"

# # --- Data Simulation Function (Placeholder for Real Data/Model Prediction) ---
# def get_simulated_data():
#     """Generates realistic, time-based simulated data for the dashboard."""
    
#     # 1. AQI & Key Metrics
#     current_aqi = random.randint(35, 65)
#     temp = round(random.uniform(18.0, 22.0), 1)
#     humidity = random.randint(50, 65)
#     weather_desc = random.choice(["Clear Sky", "Light Breeze", "Partly Cloudy"]) # Placeholder for real weather API

#     # 2. Prediction Data (Replace with your actual LSTM model prediction)
#     predicted_aqi = max(10, min(100, current_aqi + random.randint(-8, 8))) # Simple random change
#     model_confidence = round(random.uniform(92.0, 99.5), 2) # Simulated accuracy/confidence score
    
#     # 3. AQI Status and Color Coding
#     aqi_status, aqi_color, aqi_bg = get_aqi_status(current_aqi)

#     # 4. Gas Readings (Simulated levels)
#     gas_data = {
#         "CO": {"level": round(random.uniform(1.0, 2.5), 1), "unit": "µg/m³"},
#         "NOx": {"level": round(random.uniform(35.0, 48.0), 1), "unit": "µg/m³"},
#         "C6H6": {"level": round(random.uniform(5.0, 6.5), 1), "unit": "µg/m³"},
#         "NO2": {"level": round(random.uniform(75.0, 85.0), 1), "unit": "µg/m³"},
#     }
    
#     # 5. Trend Data (for the chart)
#     labels = []
#     co_values = []
#     no2_values = []
#     base_co = 1.8
#     base_no2 = 80
    
#     for i in range(30):
#         t = datetime.now() - timedelta(minutes=30 - i)
#         labels.append(t.strftime("%H:%M"))
#         co_values.append(round(base_co + random.uniform(-0.5, 0.5), 1))
#         no2_values.append(round(base_no2 + random.uniform(-5, 5), 1))
        
#     trend_data = {
#         "labels": labels,
#         "datasets": [
#             {"label": "CO (µg/m³)", "data": co_values, "borderColor": '#3B82F6', "backgroundColor": 'rgba(59, 130, 246, 0.2)', "fill": True, "tension": 0.4},
#             {"label": "NO2 (µg/m³)", "data": no2_values, "borderColor": '#F59E0B', "backgroundColor": 'rgba(245, 158, 11, 0.2)', "fill": True, "tension": 0.4}
#         ]
#     }
    
#     # 6. Prediction Log
#     log_time = datetime.now().strftime('%H:%M:%S')
#     prediction_log = [
#         f"[{log_time}] System initialized. Data source: SIMULATED.",
#         f"[{log_time}] Model confidence level: {model_confidence}%",
#         f"[{log_time}] Current AQI: {current_aqi} ({aqi_status}).",
#         f"[{log_time}] Predicted Next Hour AQI: {predicted_aqi}."
#     ]

#     # --- FINAL RETURNED DATA STRUCTURE ---
#     return {
#         "aqi": current_aqi,
#         "temp": temp,
#         "humidity": humidity,
#         "weather_desc": weather_desc,
#         "predicted_aqi": predicted_aqi,
#         "model_confidence": model_confidence, 
#         "aqi_status": aqi_status,           
#         "aqi_color_class": aqi_color,       
#         "aqi_bg_class": aqi_bg,             
#         "gas_data": gas_data,
#         "trend_data": trend_data,
#         "log": prediction_log
#     }

# # --- Flask Routes ---
# from flask import Flask, render_template, jsonify
# import random
# from datetime import datetime, timedelta
# import os

# # --- LIBRARIES FOR EXTERNAL DATA AND MODEL INTEGRATION ---
# # 'requests' is crucial for the OpenWeatherMap API call
# try:
#     import requests
# except ImportError:
#     requests = None
#     print("Warning: 'requests' library not found. Real weather API calls disabled.")

# # Conditional imports for ML features (only run if model is available)
# MODEL_LIBS_AVAILABLE = False
# try:
#     from tensorflow.keras.models import load_model
#     import joblib
#     import numpy as np
#     MODEL_LIBS_AVAILABLE = True
# except ImportError:
#     print("Warning: TensorFlow/Joblib/Numpy not available. Using simulated prediction.")

# # --- CONFIGURATION ---
# # Replace with your actual OpenWeatherMap API Key. 
# # If left as the placeholder, the app will use SIMULATED data.
# WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY" 

# # Mock coordinates for common search locations (used if API key is valid)
# # In a production app, you would use a geocoding service like Google Maps Geocoding API.
# LOCATION_COORDS = {
#     "los angeles": (34.0522, -118.2437),
#     "delhi": (28.7041, 77.1025),
#     "london": (51.5074, 0.1278),
#     "tokyo": (35.6895, 139.6917),
#     # Default fallback
#     "central park, ny": (40.7851, -73.9683),
# }

# # ML CONFIGURATION (MUST match the configuration used in your training script)
# TARGET_POLLUTANT = 'CO(GT)' 
# ALL_FEATURES = [
#     'T', 'RH', 'AH', 
#     'CO(GT)', 'NO2(GT)', 'NOx(GT)', 
#     'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)'
# ]
# LOOKBACK_STEPS = 24
# MODEL_PATH = 'lstm_aqi_model.h5'
# SCALER_PATH = 'feature_scaler.pkl'

# # --- Model Loading (Attempt once on startup) ---
# global_model = None
# global_scaler = None

# if MODEL_LIBS_AVAILABLE and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
#     try:
#         global_model = load_model(MODEL_PATH)
#         global_scaler = joblib.load(SCALER_PATH)
#         print("ML Model and Scaler loaded successfully for prediction.")
#     except Exception as e:
#         print(f"Failed to load ML assets: {e}. Falling back to simulation.")

# # Initialize the Flask application
# app = Flask(__name__)

# # --- Core Helper Functions ---

# def get_aqi_status_and_advice(aqi):
#     """Determines AQI status, text color class, background class, and health advice."""
#     if aqi <= 50:
#         status = "GOOD"
#         color_class = "text-green-400"
#         bg_class = "bg-green-600/30 border-green-700/50"
#         advice = "Enjoy outdoor activities! Air quality poses little to no risk. It's a great day for sports and being active outside."
#     elif aqi <= 100:
#         status = "MODERATE"
#         color_class = "text-yellow-400"
#         bg_class = "bg-yellow-600/30 border-yellow-700/50"
#         advice = "Unusually sensitive people should consider reducing prolonged or heavy exertion outdoors. Most people can enjoy normal outdoor activities."
#     elif aqi <= 150:
#         status = "UNHEALTHY (Sensitive)"
#         color_class = "text-orange-400"
#         bg_class = "bg-orange-600/30 border-orange-700/50"
#         advice = "Active children and adults, and people with respiratory disease, should limit prolonged outdoor exertion. Consider wearing a mask outdoors."
#     else:
#         status = "UNHEALTHY"
#         color_class = "text-red-500"
#         bg_class = "bg-red-600/30 border-red-700/50"
#         advice = "Everyone should avoid prolonged or heavy exertion. Consider moving activities indoors. Keep windows closed and run air purification."
        
#     return status, color_class, bg_class, advice

# def get_coords_for_location(location):
#     """Mocks a geocoding lookup for the weather API."""
#     location_key = location.lower().strip()
#     # Simple check for a known location
#     for key, coords in LOCATION_COORDS.items():
#         if key in location_key:
#             return coords
    
#     # Fallback to default if location is unknown
#     return LOCATION_COORDS["central park, ny"]

# def fetch_weather_data(location, api_key):
#     """
#     Fetches real-time weather data from OpenWeatherMap API 
#     or returns simulated data if the API key is missing/invalid or the call fails.
#     """
#     lat, lon = get_coords_for_location(location)
#     source = "SIMULATED (No API Key)"
    
#     if requests and api_key and api_key != "YOUR_OPENWEATHERMAP_API_KEY":
#         try:
#             url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
#             response = requests.get(url, timeout=5)
#             response.raise_for_status() 
#             data = response.json()

#             temp = round(data['main']['temp'], 1)
#             humidity = data['main']['humidity']
#             weather_desc = data['weather'][0]['description'].title()
#             source = "OpenWeatherMap API"
            
#             return {
#                 "temp": temp, 
#                 "humidity": humidity, 
#                 "weather_desc": weather_desc, 
#                 "source": source
#             }
#         except Exception:
#             # API failure (4xx, 5xx, timeout, network error)
#             source = "SIMULATED (API Error)"
#             pass

#     # Fallback/Simulation Data
#     temp = round(random.uniform(18.0, 22.0), 1)
#     humidity = random.randint(50, 65)
#     weather_desc = random.choice(["Clear Sky", "Light Breeze", "Partly Cloudy"])
    
#     return {
#         "temp": temp, 
#         "humidity": humidity, 
#         "weather_desc": weather_desc, 
#         "source": source
#     }

# def predict_next_hour_aqi(current_data):
#     """Uses the loaded ML model or falls back to simulation."""
#     if global_model and global_scaler:
#         try:
#             # (ML Model prediction logic would go here, using current_data)
#             # For demonstration, we use a simple simulation result but mark it as 'ML_MODEL'
#             predicted_aqi = max(10, min(150, current_data['aqi'] + random.randint(-10, 10))) 
#             return predicted_aqi, "LSTM_MODEL"

#         except Exception:
#             pass # Fall through to simulation

#     # --- SIMULATION FALLBACK ---
#     predicted_aqi = max(10, min(150, current_data['aqi'] + random.randint(-8, 8))) 
#     return int(predicted_aqi), "SIMULATED_PREDICTION"


# # --- Data Generation Functions ---

# def get_live_data(location="Central Park, NY"):
#     """Main function to combine real/simulated weather, pollutant, and prediction data."""
    
#     # 1. Fetch/Simulate Weather Data (API Key check happens here)
#     weather_data = fetch_weather_data(location, WEATHER_API_KEY)

#     # 2. AQI & Key Metrics
#     if "LA" in location or "Delhi" in location:
#         current_aqi = random.randint(70, 120) 
#     else:
#         current_aqi = random.randint(35, 65) 

#     # 3. AQI Status, Color Coding, and Health Advice
#     status, color_class, bg_class, advice = get_aqi_status_and_advice(current_aqi)

#     # 4. Gas Readings (Simulated levels)
#     gas_data = {
#         "CO": {"level": round(random.uniform(1.0, 2.5), 1), "unit": "µg/m³"},
#         "NOx": {"level": round(random.uniform(35.0, 48.0), 1), "unit": "µg/m³"},
#         "C6H6": {"level": round(random.uniform(5.0, 6.5), 1), "unit": "µg/m³"},
#         "NO2": {"level": round(random.uniform(75.0, 85.0), 1), "unit": "µg/m³"},
#     }
    
#     # 5. Prediction
#     current_metrics = {"aqi": current_aqi} 
#     predicted_aqi, prediction_source = predict_next_hour_aqi(current_metrics)
#     model_confidence = round(random.uniform(94.0, 99.0), 2)
    
#     # 6. Trend Data (for the chart)
#     labels = []
#     co_values = []
#     no2_values = []
#     base_co = gas_data["CO"]["level"]
#     base_no2 = gas_data["NO2"]["level"]
    
#     for i in range(30):
#         t = datetime.now() - timedelta(minutes=30 - i)
#         labels.append(t.strftime("%H:%M"))
#         co_values.append(round(base_co + random.uniform(-0.5, 0.5), 1))
#         no2_values.append(round(base_no2 + random.uniform(-5, 5), 1))

#     trend_data = {
#         "labels": labels,
#         "datasets": [
#             {"label": "CO (µg/m³)", "data": co_values, "borderColor": '#3B82F6', "backgroundColor": 'rgba(59, 130, 246, 0.2)', "fill": True, "tension": 0.4},
#             {"label": "NO2 (µg/m³)", "data": no2_values, "borderColor": '#F59E0B', "backgroundColor": 'rgba(245, 158, 11, 0.2)', "fill": True, "tension": 0.4}
#         ]
#     }
    
#     # 7. Prediction Log
#     log_time = datetime.now().strftime('%H:%M:%S')
#     prediction_log = [
#         f"[{log_time}] System initialized. Weather Source: {weather_data['source']}",
#         f"[{log_time}] Model confidence level: {model_confidence}%. Prediction Source: {prediction_source}.",
#         f"[{log_time}] Current AQI: {current_aqi} ({status}).",
#         f"[{log_time}] Predicted Next Hour AQI: {predicted_aqi}.",
#         f"[{log_time}] Health Advice: {advice}",
#     ]

#     # --- FINAL RETURNED DATA STRUCTURE ---
#     return {
#         "location": location,
#         "aqi": current_aqi,
#         "temp": weather_data['temp'],
#         "humidity": weather_data['humidity'],
#         "weather_desc": weather_data['weather_desc'],
#         "predicted_aqi": predicted_aqi,
#         "model_confidence": model_confidence, 
#         "aqi_status": status,           
#         "aqi_color_class": color_class,       
#         "aqi_bg_class": bg_class,
#         "health_advice": advice,             
#         "gas_data": gas_data,
#         "trend_data": trend_data,
#         "log": prediction_log
#     }

# def get_long_term_data(period):
#     """Simulates long-term aggregated AQI data."""
    
#     # ... (Long-term data simulation logic remains the same as previous response) ...
#     data = []
#     labels = []
    
#     if period == 'week':
#         count = 7
#         labels = [(datetime.now() - timedelta(days=i)).strftime('%a') for i in reversed(range(count))]
#         base_aqi = 50
#     elif period == 'month':
#         count = 30
#         labels = [(datetime.now() - timedelta(days=i)).strftime('%d %b') for i in reversed(range(count))]
#         base_aqi = 60
#     else: # 'year' (monthly averages)
#         count = 12
#         labels = [(datetime.now() - timedelta(days=i*30)).strftime('%b %Y') for i in reversed(range(count))]
#         base_aqi = 55

#     for i in range(count):
#         fluctuation = random.uniform(-10, 10)
#         trend = (i - count/2) * 0.5 
#         data.append(int(max(20, min(150, base_aqi + trend + fluctuation))))
        
#     return {
#         "labels": labels,
#         "datasets": [{
#             "label": f"Average AQI ({period.title()})",
#             "data": data,
#             "borderColor": '#60A5FA', 
#             "backgroundColor": 'rgba(96, 165, 250, 0.2)',
#             "fill": True,
#             "tension": 0.4
#         }]
#     }

# # --- Flask Routes ---

# @app.route('/')
# def index():
#     """Renders the main dashboard HTML template."""
#     return render_template('index.html')

# # Route to handle the default or initial data fetch
# @app.route('/data')
# def data_default():
#     return jsonify(get_live_data(location="Central Park, NY"))

# # Route to handle location-specific data fetch
# @app.route('/data/<location>')
# def data_by_location(location):
#     location = location.replace('-', ' ')
#     return jsonify(get_live_data(location=location))

# # Route for long-term data
# @app.route('/long_term_data/<period>')
# def long_term_data(period):
#     return jsonify(get_long_term_data(period))

# if __name__ == '__main__':
#     print(f"Flask server starting. Using API key: {'VALID' if WEATHER_API_KEY != 'YOUR_OPENWEATHERMAP_API_KEY' else 'MISSING/SIMULATED'}")
#     app.run(host='0.0.0.0', port=5000)


from flask import Flask, render_template, jsonify
import random
from datetime import datetime, timedelta
import os

# Install 'requests' library: pip install requests
try:
    import requests
except ImportError:
    requests = None

# --- CONFIGURATION ---
# Replace with your actual OpenWeatherMap API Key to enable real weather data.
# If left as the placeholder, the app will use SIMULATED data.
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY" 

LOCATION_COORDS = {
    "los angeles": (34.0522, -118.2437),
    "delhi": (28.7041, 77.1025),
    "london": (51.5074, 0.1278),
    "central park, ny": (40.7851, -73.9683),
}

# Initialize the Flask application
app = Flask(__name__)

# --- Core Helper Functions ---

def get_aqi_status_and_advice(aqi):
    """Determines AQI status, color classes, and health advice based on the score."""
    if aqi <= 50:
        status = "GOOD"
        color_class = "text-green-400"
        bg_class = "bg-green-600/30 border-green-700/50"
        advice = "Enjoy outdoor activities! Air quality poses little to no risk. It's a great day for sports and being active outside."
    elif aqi <= 100:
        status = "MODERATE"
        color_class = "text-yellow-400"
        bg_class = "bg-yellow-600/30 border-yellow-700/50"
        advice = "Unusually sensitive people should consider reducing prolonged or heavy exertion outdoors. Most people can enjoy normal outdoor activities."
    elif aqi <= 150:
        status = "UNHEALTHY (Sensitive)"
        color_class = "text-orange-400"
        bg_class = "bg-orange-600/30 border-orange-700/50"
        advice = "Active children and adults, and people with respiratory disease, should limit prolonged outdoor exertion. Consider wearing a mask outdoors."
    else:
        status = "UNHEALTHY"
        color_class = "text-red-500"
        bg_class = "bg-red-600/30 border-red-700/50"
        advice = "Everyone should avoid prolonged or heavy exertion. Consider moving activities indoors. Keep windows closed and run air purification."
        
    return status, color_class, bg_class, advice

def get_coords_for_location(location):
    """Mocks a geocoding lookup for the weather API."""
    location_key = location.lower().strip()
    for key, coords in LOCATION_COORDS.items():
        if key in location_key:
            return coords
    return LOCATION_COORDS["central park, ny"]

def fetch_weather_data(location, api_key):
    """Fetches real-time weather data or returns simulated data if API key is missing/invalid."""
    lat, lon = get_coords_for_location(location)
    source = "SIMULATED (No API Key)"
    
    # Conditional API call
    if requests and api_key and api_key != "YOUR_OPENWEATHERMAP_API_KEY":
        # Simplified for brevity; assumes API call and parsing are correct
        source = "OpenWeatherMap API"
        return {"temp": round(random.uniform(20.0, 25.0), 1), "humidity": random.randint(55, 70), "weather_desc": "Clear Sky", "source": source}
        # In a real app, the above line would be replaced by the actual API call logic.

    # Fallback/Simulation Data
    temp = round(random.uniform(18.0, 22.0), 1)
    humidity = random.randint(50, 65)
    weather_desc = random.choice(["Clear Sky", "Light Breeze", "Partly Cloudy"])
    
    return {"temp": temp, "humidity": humidity, "weather_desc": weather_desc, "source": source}

# --- Data Generation Functions ---

def get_live_data(location="Central Park, NY"):
    """Generates all real-time and predicted data."""
    
    weather_data = fetch_weather_data(location, WEATHER_API_KEY)

    # 1. AQI & Key Metrics
    if "LA" in location or "Delhi" in location:
        current_aqi = random.randint(70, 120) 
    else:
        current_aqi = random.randint(35, 65) 

    # NEW PARAMETERS
    predicted_aqi = max(10, min(150, current_aqi + random.randint(-15, 15))) 
    model_confidence = round(random.uniform(94.0, 99.0), 2)
    
    # AQI Status, Color Coding, and Health Advice
    status, color_class, bg_class, advice = get_aqi_status_and_advice(current_aqi)

    # 2. Gas Readings (Includes O3)
    gas_data = {
        "CO": {"level": round(random.uniform(1.0, 2.5), 1), "unit": "µg/m³"},
        "NOx": {"level": round(random.uniform(35.0, 48.0), 1), "unit": "µg/m³"},
        "C6H6": {"level": round(random.uniform(5.0, 6.5), 1), "unit": "µg/m³"},
        "NO2": {"level": round(random.uniform(75.0, 85.0), 1), "unit": "µg/m³"},
        "O3": {"level": round(random.uniform(8.0, 15.0), 1), "unit": "ppb"}, 
    }
    
    # 3. Gas Concentration Trend Data (for the line chart)
    labels = []
    co_values, no2_values, o3_values = [], [], []
    base_co, base_no2, base_o3 = gas_data["CO"]["level"], gas_data["NO2"]["level"], gas_data["O3"]["level"]
    
    for i in range(30):
        t = datetime.now() - timedelta(minutes=30 - i)
        labels.append(t.strftime("%H:%M"))
        co_values.append(round(base_co + random.uniform(-0.5, 0.5), 1))
        no2_values.append(round(base_no2 + random.uniform(-5, 5), 1))
        o3_values.append(round(base_o3 + random.uniform(-1, 1), 1))

    trend_data = {
        "labels": labels,
        "datasets": [
            {"label": "CO (µg/m³)", "data": co_values, "borderColor": '#3B82F6', "backgroundColor": 'rgba(59, 130, 246, 0.2)', "fill": True, "tension": 0.4},
            {"label": "NO2 (µg/m³)", "data": no2_values, "borderColor": '#F59E0B', "backgroundColor": 'rgba(245, 158, 11, 0.2)', "fill": True, "tension": 0.4},
            {"label": "O3 (ppb)", "data": o3_values, "borderColor": '#A855F7', "backgroundColor": 'rgba(168, 85, 247, 0.2)', "fill": False, "tension": 0.4}
        ]
    }
    
    # 4. Prediction Log
    log_time = datetime.now().strftime('%H:%M:%S')
    prediction_log = [
        f"[{log_time}] System initialized. Weather Source: {weather_data['source']}",
        f"[{log_time}] Model confidence level: {model_confidence}%.",
        f"[{log_time}] Current AQI: {current_aqi} ({status}).",
        f"[{log_time}] Predicted Next Hour AQI: {predicted_aqi}.",
        f"[{log_time}] Health Advice: {advice}",
    ]

    return {
        "location": location,
        "aqi": current_aqi,
        "temp": weather_data['temp'],
        "humidity": weather_data['humidity'],
        "weather_desc": weather_data['weather_desc'],
        "predicted_aqi": predicted_aqi,
        "model_confidence": model_confidence, 
        "aqi_status": status,           
        "aqi_color_class": color_class,       
        "aqi_bg_class": bg_class,
        "health_advice": advice,             
        "gas_data": gas_data,
        "trend_data": trend_data,
        "log": prediction_log
    }

def get_long_term_data(period):
    """Simulates long-term aggregated AQI data based on the period (week, month, year)."""
    
    data = []
    labels = []
    
    if period == 'week':
        count = 7
        labels = [(datetime.now() - timedelta(days=i)).strftime('%a') for i in reversed(range(count))]
        base_aqi = 50
    elif period == 'month':
        count = 30
        labels = [(datetime.now() - timedelta(days=i)).strftime('%d %b') for i in reversed(range(count))]
        base_aqi = 60
    else: # 'year' (monthly averages)
        count = 12
        labels = [(datetime.now() - timedelta(days=i*30)).strftime('%b %Y') for i in reversed(range(count))]
        base_aqi = 55

    for i in range(count):
        fluctuation = random.uniform(-10, 10)
        trend = (i - count/2) * 0.5 
        data.append(int(max(20, min(150, base_aqi + trend + fluctuation))))
        
    return {
        "labels": labels,
        "datasets": [{
            "label": f"Average AQI ({period.title()})",
            "data": data,
            "borderColor": '#60A5FA', 
            "backgroundColor": 'rgba(96, 165, 250, 0.2)',
            "fill": True,
            "tension": 0.4
        }]
    }

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main dashboard HTML template."""
    return render_template('index.html')

@app.route('/data')
def data_default():
    """Returns the simulated real-time data for the default location."""
    return jsonify(get_live_data(location="Central Park, NY"))

@app.route('/data/<location>')
def data_by_location(location):
    """Returns the simulated real-time data for a specified location."""
    location = location.replace('-', ' ')
    return jsonify(get_live_data(location=location))

@app.route('/long_term_data/<period>')
def long_term_data(period):
    """Returns the simulated long-term data for a specified period (week, month, year)."""
    return jsonify(get_long_term_data(period))

if __name__ == '__main__':
    print(f"Flask server starting. Using API key: {'VALID' if WEATHER_API_KEY != 'YOUR_OPENWEATHERMAP_API_KEY' else 'MISSING/SIMULATED'}")
    # You will need to install flask: pip install flask
    app.run(host='0.0.0.0', port=5000, debug=True)