# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load Dataset
df = pd.read_csv("loan_data_encoded.csv")

# Display Dataset Information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Drop Loan_ID (Identifier)
if 'Loan_ID' in df.columns:
    df = df.drop('Loan_ID', axis=1)

# Features and Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", round(accuracy * 100, 2), "%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# Sample Prediction
sample = X_test.iloc[[0]]

prediction = model.predict(sample)

print("\nSample Prediction:")

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")