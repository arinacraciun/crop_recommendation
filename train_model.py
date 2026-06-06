import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.metrics import confusion_matrix
import joblib

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# 1. Load Data
df = pd.read_csv('data/Crop_recommendation.csv')

# 2. Preprocessing: Separate features and encode target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Train-Test Split & Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# LightGBM is highly optimized for tabular data and fast inference
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 4. Save the Model and the Label Encoder
joblib.dump(model, 'models/lgbm_model.joblib')
joblib.dump(le, 'models/label_encoder.joblib')

# 5. Pre-compute Global Feature Importance for Tab 2
# Saving this as a CSV avoids recalculating it every time the dashboard loads
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=True)

importance_df.to_csv('models/global_importance.csv', index=False)

# 6. Pre-compute Ideal Crop Conditions for the Radar Chart
# We calculate the mean of every feature for each specific crop
crop_means_df = df.groupby('label').mean().reset_index()
crop_means_df.to_csv('models/crop_means.csv', index=False)

# --- NEW: Artifacts for Tab 3 Diagnostics ---

# Get Test Set Predictions
y_pred = model.predict(X_test)

# Save normalized Confusion Matrix (Percentages, not raw counts)
cm = confusion_matrix(y_test, y_pred, normalize='true')
joblib.dump(cm, 'models/test_confusion_matrix.joblib')

# Pre-compute Global Feature Importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=True)
importance_df.to_csv('models/global_importance.csv', index=False)

# Save test-set target data and predictions for reporting
np.save('models/test_targets.npy', y_test)
np.save('models/test_preds.npy', y_pred)

print("Diagnostic artifacts saved.")

print("Model, Encoder, and Global Importance saved successfully.")