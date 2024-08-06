# Prodigy ML Task 1

This project demonstrates data preprocessing and linear regression modeling using pandas, numpy, and scikit-learn. It includes techniques for handling null values, feature scaling, and model evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project involves:
- Loading and cleaning datasets
- Handling null values in numeric and non-numeric columns
- Scaling features
- Training a linear regression model
- Evaluating the model's performance

## Installation

To run this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- scikit-learn

You can install the necessary packages using:

```bash
pip install pandas numpy scikit-learn
Data Preprocessing
The script performs the following preprocessing steps:

Fills null values in numeric columns with the mean
Fills null values in non-numeric columns with the most frequent value (mode)
Normalizes the feature data using StandardScaler
Model Training and Evaluation
The linear regression model is trained using the preprocessed training data. The model's performance is evaluated using the following metrics:

Mean Squared Error (MSE)
R-squared (R2)
Mean Absolute Error (MAE)
Explained Variance Score
Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

License
This project is licensed under the MIT License. See the LICENSE file for detail
