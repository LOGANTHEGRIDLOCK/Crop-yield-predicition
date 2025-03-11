import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib

# Load dataset
file_path = 'data/cropYield.csv'  # Ensure correct path
crop_yield_data = pd.read_csv(file_path)

# Drop unnecessary column and remove duplicates
if 'Unnamed: 0' in crop_yield_data.columns:
    crop_yield_data = crop_yield_data.drop(columns=['Unnamed: 0'])
crop_yield_data = crop_yield_data.drop_duplicates()

# Label Encoding for Categorical Columns
label_encoders = {}
for col in ['Area', 'Item']:
    le = LabelEncoder()
    crop_yield_data[col] = le.fit_transform(crop_yield_data[col])
    label_encoders[col] = le  # Store encoder for later use

# Feature Engineering
crop_yield_data['rainfall_temp_interaction'] = (
    crop_yield_data['average_rain_fall_mm_per_year'] * crop_yield_data['avg_temp']
)
crop_yield_data['log_pesticides'] = np.log1p(crop_yield_data['pesticides_tonnes'])

# Splitting Features and Target
X = crop_yield_data.drop(columns=['hg/ha_yield'])
y = crop_yield_data['hg/ha_yield']

# Scaling the features
scaler = StandardScaler()
X_Scaled_Cols = ['average_rain_fall_mm_per_year', 'avg_temp', 'rainfall_temp_interaction', 'log_pesticides']
X[X_Scaled_Cols] = scaler.fit_transform(X[X_Scaled_Cols])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with hyperparameter tuning
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
param_grid_svm = {
    'C': [0.1, 1],
    'kernel': ['linear', 'rbf']
}

# Initialize models
rf_model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='r2')
gb_model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=5, scoring='r2')
svm_model = GridSearchCV(SVR(), param_grid_svm, cv=5, scoring='r2')

# Train models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Get best models
best_rf = rf_model.best_estimator_
best_gb = gb_model.best_estimator_
best_svm = svm_model.best_estimator_

# Evaluate models
models = {'RandomForest': best_rf, 'GradientBoosting': best_gb, 'SVM': best_svm}
results = {}
for name, model in models.items():
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    results[name] = {'R2 Train': r2_train, 'R2 Test': r2_test, 'MAE': mae_test}

# Print results
print(results)

# Save the best model (choose the one with highest R2 Test score)
best_model_name = max(results, key=lambda k: results[k]['R2 Test'])
best_model = models[best_model_name]
joblib.dump(best_model, "optimized_crop_yield_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"Best model saved: {best_model_name} -> optimized_crop_yield_model.pkl")
