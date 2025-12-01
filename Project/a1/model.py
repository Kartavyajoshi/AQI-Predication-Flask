import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib # Library for saving the scaler object
import os # For checking file paths

# --- Configuration ---
# Set the desired pollutant target. The model will be trained for this single pollutant.
TARGET_POLLUTANT = 'CO(GT)' 
# List of ALL features (pollutants and auxiliary data) to be used as input.
# Assuming a standard comprehensive set of features often found in air quality datasets.
ALL_FEATURES = [
    'T', 'RH', 'AH', 
    'CO(GT)', 'NO2(GT)', 'NOx(GT)', 
    'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)'
]
# Number of previous hours to use to predict the next hour
LOOKBACK_STEPS = 24 
# The placeholder value used in the dataset for missing data
MISSING_VALUE = -200 
# File paths for saving the trained model and scaler
MODEL_PATH = 'lstm_aqi_model.h5'
SCALER_PATH = 'feature_scaler.pkl'


def preprocess_data(df):
    """
    Cleans the data, handles missing values, and selects ALL defined features.
    The script assumes these feature columns exist in the input dataframe.
    """
    print("Starting data preprocessing and feature selection...")

    # 1. Handle missing values/placeholders: Replace -200 with NaN
    df.replace(MISSING_VALUE, np.nan, inplace=True)
    
    # 2. Select only the specified ALL_FEATURES
    # Filter out columns that do not exist if the mock data is simpler
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    
    # Check if the target pollutant is in the available features
    if TARGET_POLLUTANT not in available_features:
        print(f"Error: Target pollutant '{TARGET_POLLUTANT}' not found in data.")
        return None

    df = df[available_features].copy()

    # 3. Time-Series Imputation: Use 'ffill' (forward fill) followed by 'bfill' (backward fill)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # 4. Final check to ensure no NaNs remain (e.g., if first/last values were NaN)
    if df.isnull().any().any():
        print("Warning: Dropping rows with remaining NaNs after imputation.")
        df.dropna(inplace=True)
    
    print(f"Data shape after cleaning: {df.shape}")
    print(f"Features used in the model: {df.columns.tolist()}")
    return df

def create_sequences(data, lookback, target_column_index):
    """
    Converts the time-series data (multi-dimensional) into sequences (X) and 
    the next target value (y) for LSTM training.
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        # Input sequence (lookback history of ALL features)
        X.append(data[i:(i + lookback)]) 
        # Target value (the pollutant concentration at the next hour)
        y.append(data[i + lookback][target_column_index]) 
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Defines and compiles the LSTM model architecture.
    """
    model = Sequential([
        # The input shape now uses all features (X_train.shape[2])
        LSTM(units=128, activation='tanh', input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        
        LSTM(units=64, activation='tanh'),
        Dropout(0.3),
        
        # Output layer: 1 neuron for the single predicted pollutant concentration
        Dense(units=1)
    ])
    
    # Using Adam optimizer and Mean Squared Error (MSE) loss for regression
    model.compile(optimizer='adam', loss='mse')
    print("LSTM Model Summary:")
    model.summary()
    return model

def main():
    # --- 1. Load Data (Using a mock structure, replace with your actual file loading) ---
    print("Loading data... (Assuming 'AirQualityUCI.csv' structure)")
    try:
        data_path = 'Project/AirQualityUCI.csv' # <-- REPLACE THIS
        try:
             # Try to read the file assuming a specific separator (if the user has a UCI-like file)
            df = pd.read_csv(data_path, sep=';', decimal=',', na_values=['', 'NA']) 
        except FileNotFoundError:
            print(f"File not found at {data_path}. Creating mock data with ALL features for demonstration.")
            N_SAMPLES = 9357
            data = {
                'T': np.random.uniform(10, 35, N_SAMPLES),
                'RH': np.random.uniform(20, 90, N_SAMPLES),
                'AH': np.random.uniform(0.5, 2.5, N_SAMPLES),
                'CO(GT)': np.random.uniform(0.1, 10, N_SAMPLES),
                'NO2(GT)': np.random.uniform(10, 150, N_SAMPLES),
                'NOx(GT)': np.random.uniform(50, 600, N_SAMPLES),
                'PT08.S1(CO)': np.random.uniform(800, 1500, N_SAMPLES),
                'PT08.S2(NMHC)': np.random.uniform(500, 1500, N_SAMPLES),
                'PT08.S3(NOx)': np.random.uniform(400, 2000, N_SAMPLES),
                'PT08.S4(NO2)': np.random.uniform(800, 2500, N_SAMPLES),
                'PT08.S5(O3)': np.random.uniform(800, 2500, N_SAMPLES),
            }
            # Simulate some missing data only for the target pollutant
            data[TARGET_POLLUTANT][100:150] = MISSING_VALUE
            df = pd.DataFrame(data)

    except Exception as e:
        print(f"Error loading data: {e}. Please check your file path and format.")
        return

    # --- 2. Preprocess Data (Now uses all relevant columns) ---
    df_processed = preprocess_data(df)
    if df_processed is None:
        return
    
    # Get the index of the target pollutant column in the processed dataframe
    target_column_index = df_processed.columns.get_loc(TARGET_POLLUTANT)
    
    # --- 3. Scaling (Uses ALL features) ---
    # The scaler must be fitted on all features for correct inverse transformation later.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_processed)
    
    # --- 4. Sequence Generation (X, y) ---
    # X now contains sequences of all features. y contains the next step's target pollutant.
    X, y = create_sequences(scaled_data, LOOKBACK_STEPS, target_column_index)

    # --- 5. Train/Test Split (Time-Series Split) ---
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    # --- 6. Model Training ---
    # Input shape is (LOOKBACK_STEPS, number_of_features)
    input_shape = (X_train.shape[1], X_train.shape[2]) 
    model = build_lstm_model(input_shape)

    # Use Early Stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)

    print("\nStarting model training...")
    history = model.fit(
        X_train, 
        y_train, 
        epochs=100, # Increased epochs for better training
        batch_size=64, 
        validation_split=0.1, 
        callbacks=[early_stop],
        verbose=1
    )
    
    # --- 7. Model Persistence: Save Model and Scaler ---
    try:
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"\nModel saved successfully to {MODEL_PATH}.")
        print(f"Scaler saved successfully to {SCALER_PATH}. This is crucial for future predictions!")
    except Exception as e:
        print(f"Could not save model/scaler: {e}")

    # --- 8. Evaluation and Prediction ---
    print("\nEvaluating model performance on test set...")
    mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Set Mean Squared Error (MSE) for {TARGET_POLLUTANT}: {mse:.4f}")

    # Get predictions
    predicted_scaled = model.predict(X_test)
    
    # Invert the scaling for the predicted values
    # We need a temporary zero array of shape (N_samples, N_features)
    temp_array = np.zeros((len(predicted_scaled), scaled_data.shape[1])) 
    temp_array[:, target_column_index] = predicted_scaled.flatten() 
    predicted_concentrations = scaler.inverse_transform(temp_array)[:, target_column_index]
    
    # Also invert transform the true values for comparison
    temp_array_true = np.zeros((len(y_test), scaled_data.shape[1]))
    temp_array_true[:, target_column_index] = y_test
    true_concentrations = scaler.inverse_transform(temp_array_true)[:, target_column_index]
    
    print(f"\nExample Predictions (Predicted vs. True {TARGET_POLLUTANT} Concentration):")
    for i in range(5):
        print(f"Time {i+1}: Predicted = {predicted_concentrations[i]:.2f}, True = {true_concentrations[i]:.2f}")
    
    # --- 9. AQI Calculation Step (Conceptual) ---
    print("\n--- Final Step: AQI Calculation ---")
    print("Use the predicted concentrations and the saved scaler/model for deployment.")
    print("Remember to re-train or update the model periodically with new data to maintain accuracy.")
    print("Example: predicted_concentrations[i] -> (AQI Formula) -> IAQI -> Final AQI")


if __name__ == "__main__":
    main()