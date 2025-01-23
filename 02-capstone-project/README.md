## Uber Ride Price Prediction

![image](https://github.com/user-attachments/assets/eae7e5ad-f20d-41b5-a53d-c5eb313f5928)

### Overview

The Uber Ride Price Prediction project aims to develop a machine learning model capable of accurately predicting the price of Uber rides. By utilizing historical ride data and advanced regression techniques, the project seeks to provide insights that can optimize costs for both passengers and drivers, ultimately enhancing the efficiency of ride-hailing services.

### Problem Description

The cost of Uber rides can vary based on numerous factors, including distance, time of day, weather conditions, and surge pricing. Accurately forecasting ride prices is critical for:

- Enabling passengers to better plan their transportation expenses.
- Helping drivers optimize earnings by identifying high-demand scenarios.
- Improving Uber’s dynamic pricing mechanisms.

### Objective

The primary objective is to build a predictive model using machine learning techniques that can estimate the price of a ride based on relevant input features. The pipeline includes exploratory data analysis (EDA), feature engineering, hyperparameter tuning, and model evaluation to ensure the solution’s robustness and accuracy.

### Dataset

The dataset used for this project was sourced from Kaggle’s rideshare dataset. Key attributes include:

- Ride Information: Source, destination, cab type, distance.
- Time Information: Hour, day, timestamp.
- Weather Data: Temperature, humidity, visibility, wind speed.
- Pricing Details: Price and surge multiplier.

### Dataset Preprocessing

1. Removed irrelevant or highly correlated columns.
2. Imputed missing values for key features.
3. Normalized numerical columns and one-hot encoded categorical features.
4. Extracted temporal and interaction features to enrich the dataset.

### Project Workflow

#### 1. Exploratory Data Analysis (EDA)
EDA was conducted to:
- Identify missing values.
- Understand the distribution of key features.
- Examine relationships between features and the target variable (‘price’).

Key findings:

- Distance and surge multiplier are strongly correlated with ride prices.
- Outliers in pricing data were identified and handled.

#### 2. Feature Engineering

New features were generated to enhance the model’s performance:

- Temporal Features: Day of the week, hour of the day.
- Interaction Features: Distance multiplied by surge multiplier.

#### 3. Model Training
Models Trained:

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Hyperparameter Tuning:

- RandomizedSearchCV: Used for coarse hyperparameter tuning.
- GridSearchCV: Refined hyperparameter optimization.

Metrics:

- Root Mean Squared Error (RMSE) was used to evaluate model performance.

#### 4. Deployment

The final model was packaged and deployed as a REST API using Flask. The deployment pipeline includes a Docker container with Kubernetes configuration

### Key Files

1. train.py

- Script for training the model.
- Handles data preprocessing, feature engineering, and model evaluation.
- Saves the trained model and vectorizer as a binary file (price_prediction.bin).

2. predict.py

- Flask-based REST API for predicting ride prices.
- Accepts JSON input with ride features and returns the predicted price.

3. deployment.yaml

- Kubernetes Deployment configuration for running the Flask app in a scalable manner.
- Specifies resource limits, container port, and replica count.

4. service.yaml

- Kubernetes Service configuration to expose the Flask app externally.
- Configures a NodePort for accessing the API.

5. requirements.txt

Lists all Python dependencies required for the project, including:
- scikit-learn
- xgboost
- Flask
- numpy, pandas, etc.

6. columns_attributes.json

- Defines the metadata for each feature, including type, range, and categories.
- Facilitates data validation and preprocessing consistency.

7. .github/workflows/python-ci-cd.yml

- GitHub Actions CI/CD pipeline for:
- Dependency installation.
- Running automated tests.

8. test_predict.py

Contains unit tests for the prediction API to validate functionality and accuracy.

### Usage Instructions
Run the following commands to test the model locally:

1. Local Testing

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Start the API
python predict.py

2. API Usage

Send a POST request to the /predict endpoint with the following JSON payload:

{
  "distance": 3.5,
  "surge_multiplier": 1.2,
  "latitude": 42.36,
  "longitude": -71.06,
  "temperature": 40,
  "humidity": 0.85,
  "source": "Boston University",
  "destination": "North Station",
  "cab_type": "UberX",
  "hour": 14,
  "day": 5
}






