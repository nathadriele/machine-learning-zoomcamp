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

Develop a robust machine learning model that predicts the price of an Uber ride based on input features such as distance, time of day, weather conditions, and more. This involves preprocessing data, conducting exploratory data analysis (EDA), feature engineering, model training, and deployment.

### Dataset

The dataset used for this project was sourced from Kaggle’s rideshare dataset. Key attributes include:

- Ride Information: Source, destination, cab type, distance.
- Time Information: Hour, day, timestamp.
- Weather Data: Temperature, humidity, visibility, wind speed.
- Pricing Details: Ride price and surge multiplier.

### Dataset Preprocessing

1. Column Cleaning: Removed irrelevant and highly correlated columns.
2. Missing Value Handling: Imputed missing values or dropped rows where appropriate.
3. Feature Engineering: Created interaction terms and temporal features.
4. Normalization: Scaled numerical data and one-hot encoded categorical variables.

### Project Workflow

#### 1. Exploratory Data Analysis (EDA)

EDA provided key insights into the dataset, which were critical for feature selection and engineering:

- Correlation Analysis: Identified strong relationships between distance, surge multiplier, and price.
- Outlier Detection: Detected and handled anomalies in the pricing data.
- Temporal Trends: Observed price fluctuations based on the time of day.
- Geospatial Patterns: Visualized ride distribution using heatmaps.

### Visualizations




### Model Development

#### Models Tested

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor (Best Performing)

#### Feature Engineering

- Interaction Features: Combined distance and surge multiplier.
- Temporal Features: Added day of the week and weekend flags.

#### Training Pipeline

- Split data into training and testing sets.
- Applied DictVectorizer to transform categorical variables.
- Trained and optimized models using hyperparameter tuning.
- Evaluated models using RMSE (Root Mean Squared Error).

### Deployment

The final model was deployed as a REST API using Flask, containerized with Docker, and orchestrated using Kubernetes.

#### Deployment Architecture

- Flask API: Handles requests and predictions.
- Docker: Packages the application for portability.
- Kubernetes: Manages scaling and availability

### Usage Instructions

### Local Testing

#### Install dependencies
pip install -r requirements.txt

#### Train the model
python train.py

#### Start the API
python predict.py

### Key Files

- `train.py`: Script for data preprocessing, model training, and evaluation.
- `predict.py`: Flask application for serving predictions.
- `deployment.yaml`: Kubernetes deployment configuration.
- `service.yaml`: Kubernetes service configuration.
- `columns_attributes.json`: Metadata for input features.
- `requirements.txt`: Python dependencies.
- `pipeline.joblib`: Serialized model pipeline.

.github/workflows/python-ci-cd.yml: CI/CD pipeline for automated testing

### API Usage

Send a POST request to the /predict endpoint with a JSON payload:

```
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
```

**Response:**

```
{
  "predicted_price": 12.34
}
```

### CI/CD Pipeline

The project includes a GitHub Actions pipeline for:

- Installing dependencies.
- Running automated tests.
- Deploying the application.

### Future Improvements

- Real-Time Data Integration: Incorporate live weather and traffic data.
- Deep Learning Models: Explore neural networks for enhanced accuracy.
- User Interface: Develop a front-end for user-friendly predictions.

### Conclusion

This foundational project sought to demonstrate the complete pipeline for predicting Uber ride prices using machine learning techniques. Through detailed data preprocessing, insightful exploratory analysis, and rigorous model training, we achieved an efficient and deployable prediction system. This project is part of the practical lessons in the MLOps Zoomcamp course, designed to the end-to-end workflow of machine learning projects.

### Contribution

Contributions are welcome!

Thank you for reading and using the Uber Ride Price Prediction repository, my friend!
