# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:30:14 2024
bank-marketing-campaign-data
@author: rober ugalde

you will find the following variables:

age. Age of customer (numeric)
job. Type of job (categorical)
marital. Marital status (categorical)
education. Level of education (categorical)
default. Do you currently have credit (categorical)
housing. Do you have a housing loan (categorical)
loan. Do you have a personal loan? (categorical)
contact. Type of contact communication (categorical)
month. Last month in which you have been contacted (categorical)
day_of_week. Last day on which you have been contacted (categorical)
duration. Duration of previous contact in seconds (numeric)
campaign. Number of contacts made during this campaign to the customer (numeric)
pdays. Number of days that elapsed since the last campaign until the customer was contacted (numeric)
previous. Number of contacts made during the previous campaign to the customer (numeric)
poutcome. Result of the previous marketing campaign (categorical)
emp.var.rate. Employment variation rate. Quarterly indicator (numeric)
cons.price.idx. Consumer price index. Monthly indicator (numeric)
cons.conf.idx. Consumer confidence index. Monthly indicator (numeric)
euribor3m. EURIBOR 3-month rate. Daily indicator (numeric)
nr.employed. Number of employees. Quarterly indicator (numeric)
y. TARGET. Whether the customer takes out a long-term deposit or not (categorical)
Step 2: Perform a full EDA
This second step is vital to ensure that we keep the variables that are strictly necessary and eliminate those that are not relevant or do not provide information. Use the example Notebook we worked on and adapt it to this use case.

Be sure to conveniently divide the data set into train and test as we have seen in previous lessons.

Step 3: Build a logistic regression model
You do not need to optimize the hyperparameters. Start by using a default definition and improve it in the next step.


"""


import pandas as pd

# Load the dataset from the provided link
url = "https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv"
data = pd.read_csv(url)

# Display the first few rows to understand the structure
print(data.head())


# Inspect basic information
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Summary statistics for numeric columns
print(data.describe())

# Check the distribution of the target variable
print(data['y'].value_counts())

# Convert categorical variables into dummy variables
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                       'contact', 'month', 'day_of_week', 'poutcome']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Map target variable to binary
data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Split the dataset into features (X) and target (y)
X = data.drop('y', axis=1)
y = data['y']

# Split data into training and test sets (70% train, 30% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Create the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))


xxxx


from sklearn.model_selection import train_test_split

# Features (X) and target (y)
X = data.drop('y', axis=1)  # Ensure 'y' is the target column in your dataset
y = data['y']  # Target variable

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print(data.columns)


xxx


import pandas as pd

# Load the dataset with the correct delimiter
url = "https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv"
data = pd.read_csv(url, delimiter=";")

# Verify the columns are now correctly parsed
print(data.columns)



# Convert target column 'y' to binary values
data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# List of categorical columns to encode
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                       'contact', 'month', 'day_of_week', 'poutcome']

# One-hot encoding for categorical variables
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)



from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = data.drop('y', axis=1)
y = data['y']

# Split data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verify the split
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize and train the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))




xxxxx  complete code 


# Import necessary libraries
import pandas as pd
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

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))



xxxxx  with charts


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
