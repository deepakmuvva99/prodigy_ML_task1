import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler

# Load the datasets
trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")

# Define features and target
features = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']
target = ['SalePrice']

# Separate numeric and non-numeric columns
numeric_cols = trainData.select_dtypes(include=[np.number]).columns
non_numeric_cols = trainData.select_dtypes(exclude=[np.number]).columns

# Fill null values in numeric columns with the mean
trainData[numeric_cols] = trainData[numeric_cols].fillna(trainData[numeric_cols].mean())

# Fill null values in non-numeric columns with the most frequent value (mode)
for col in non_numeric_cols:
    trainData[col] = trainData[col].fillna(trainData[col].mode()[0])

# Separate numeric and non-numeric columns in test data
numeric_cols_test = testData.select_dtypes(include=[np.number]).columns
non_numeric_cols_test = testData.select_dtypes(exclude=[np.number]).columns

# Fill null values in numeric columns with the mean
testData[numeric_cols_test] = testData[numeric_cols_test].fillna(testData[numeric_cols_test].mean())

# Fill null values in non-numeric columns with the most frequent value (mode)
for col in non_numeric_cols_test:
    testData[col] = testData[col].fillna(testData[col].mode()[0])

# Prepare feature and target variables
x_train = trainData[features]
y_train = trainData[target]

x_test = testData[features]

# Normalize the feature data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_train_pred = model.predict(x_train)  # Predictions on training data
y_test_pred = model.predict(x_test)    # Predictions on test data

# Evaluate the model on the training data
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
explained_var_train = explained_variance_score(y_train, y_train_pred)

print("Training Data Evaluation:")
print("Mean Squared Error (MSE):", mse_train)
print("R-squared (R2):", r2_train)
print("Mean Absolute Error (MAE):", mae_train)
print("Explained Variance Score:", explained_var_train)


