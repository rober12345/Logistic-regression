from utils import db_connect
engine = db_connect()

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset with the correct delimiter
url = "https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv"
data = pd.read_csv(url, delimiter=";")

# Convert the target column 'y' to binary values
data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# List of categorical columns to encode
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                       'contact', 'month', 'day_of_week', 'poutcome']

# One-hot encoding for categorical variables
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Define features (X) and target (y)
X = data.drop('y', axis=1)
y = data['y']

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for the positive class

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Plot the regression chart
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_prob, alpha=0.6, label="Predicted Probability")
plt.xlabel("Actual Class (y_test)")
plt.ylabel("Predicted Probability (y_prob)")
plt.title("Logistic Regression: Predicted Probability vs Actual Class")
plt.axhline(0.5, color='red', linestyle='--', label="Decision Threshold (0.5)")
plt.legend()
plt.grid()
plt.show()
