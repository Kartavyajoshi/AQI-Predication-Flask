import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib # Used to save the trained model
import numpy as np

DATASET_PATH = 'AirQualityUCI.csv'
data = pd.read_csv(DATASET_PATH,sep=";",decimal=",")   # Replace , --> .  and seperator ;
print(f"Dataset Shape: {data.shape}")
print("First 5 rows of the dataset:")
print(data.head(5))

# Check count of nulls per column
df.isnull().sum()

# --- 1.1 Data Cleaning (Basic Example) ---
# Handle missing values by dropping rows with any NaN values. 
# For a real project, you might use imputation (filling missing data).

#   # Target_AQI is calculated based on pollutant levels (a synthetic relationship)
# data['Target_AQI'] = 10 + (data['CO'] * 5) + (data['NOx'] * 0.5) + (data['C6H6'] * 2) + np.random.randint(-10, 10, n_samples)
# data['Target_AQI'] = np.clip(data['Target_AQI'], 10, 150).astype(int) # Limit AQI to a realistic range

# df = pd.DataFrame(data)
    
    
# # --- 2. Define Features and Target ---
# # Features (X) are the input columns used for prediction.
# # Target (y) is the output column we want to predict (e.g., AQI).
# feature_cols = ['CO', 'NOx', 'C6H6', 'Temp', 'Humidity']
# target_col = 'Target_AQI'

# if not all(col in df.columns for col in feature_cols + [target_col]):
#     print("\nFATAL ERROR: Your dataset is missing one or more required columns:")
#     print(f"Expected Features: {feature_cols}")
#     print(f"Expected Target: {target_col}")
#     print("Please check your CSV file column names and try again.")
#     # Exit gracefully if columns are missing
#     exit()

# X = df[feature_cols]
# y = df[target_col]

# # --- 3. Split Data for Training and Testing ---
# # Split the data into training (80%) and testing (20%) sets.
# # Training data is used to teach the model; test data is used for evaluation.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("\n" + "=" * 40)
# print("STARTING MODEL TRAINING")
# print(f"Training set size: {len(X_train)} samples")
# print(f"Testing set size: {len(X_test)} samples")
# print("=" * 40)

# # --- 4. Initialize and Train the Model ---
# # We use a Random Forest Regressor, a robust model for regression tasks.
# model = RandomForestRegressor(
#     n_estimators=100,      # Number of decision trees in the forest
#     random_state=42,       
#     n_jobs=-1,             # Use all available CPU cores
#     max_depth=10           # Limit the depth to prevent overfitting
# )
# print("Initializing Random Forest Regressor...")
# model.fit(X_train, y_train)
# print("Model training completed successfully.")

# # --- 5. Evaluate the Model ---
# y_pred = model.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("\n" + "=" * 40)
# print(f"MODEL EVALUATION RESULTS (on 20% test data)")
# print(f"Mean Squared Error (MSE): {mse:.2f}")
# print(f"R-squared (R2 Score): {r2:.4f}")
# print("-" * 40)
# print("Interpretation:")
# print(f"  MSE: Measures the average squared difference between predicted and actual values (lower is better).")
# print(f"  R2 Score: Represents the proportion of variance explained by the model (closer to 1.0 is better).")


# # --- 6. Save the Trained Model ---
# model_filename = 'air_quality_model.pkl'
# joblib.dump(model, model_filename)
# print("=" * 40)
# print(f"Model successfully saved as '{model_filename}'.")
# print("This file can now be loaded by 'app.py' for real-time predictions.")