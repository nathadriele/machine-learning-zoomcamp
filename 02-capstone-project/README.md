## Uber and Lyft Ride Price Prediction

![image](https://github.com/user-attachments/assets/eae7e5ad-f20d-41b5-a53d-c5eb313f5928)

### Overview

The Uber and Lyft Ride Price Prediction project aims to develop a machine learning model capable of accurately predicting the price of Uber rides. By utilizing historical ride data and advanced regression techniques, the project seeks to provide insights that can optimize costs for both passengers and drivers, ultimately enhancing the efficiency of ride-hailing services.

### Problem Description

The cost of Uber rides can vary based on numerous factors, including distance, time of day, weather conditions, and surge pricing. Accurately forecasting ride prices is critical for:

- Enabling passengers to better plan their transportation expenses.
- Helping drivers optimize earnings by identifying high-demand scenarios.
- Improving Uber’s dynamic pricing mechanisms.

### Objective

Develop a robust machine learning model that predicts the price of an Uber ride based on input features such as distance, time of day, weather conditions, and more. This involves preprocessing data, conducting exploratory data analysis (EDA), feature engineering, model training, and deployment.

### Dataset

![image](https://github.com/user-attachments/assets/a0406514-9f22-4418-a13f-54be80248fdd)

The dataset used for this project was sourced from Kaggle’s rideshare dataset. Key attributes include:

- Ride Information: Source, destination, cab type, distance.
- Time Information: Hour, day, timestamp.
- Weather Data: Temperature, humidity, visibility, wind speed.
- Pricing Details: Ride price and surge multiplier.

### Project Structure

![image](https://github.com/user-attachments/assets/9c7c192c-eb19-44ad-a949-e8a958ef73d5)

     - ci_cd/: Contains the configuration files for Continuous Integration and Continuous Deployment (CI/CD) pipelines. For example, python-ci-cd.yml is used to automate testing and deployment workflows.
     - configs/: Stores configuration files such as columns_attributes.json, which include important settings or metadata required by the project.
     - data/: Contains datasets used in the project, such as rideshare_kaggle.csv. This directory is used to manage both raw and processed data.
     - deployments/: Includes files related to deploying the project, such as:
     - Dockerfile: Instructions to containerize the application.
     - deployment.yaml and service.yaml: Kubernetes deployment and service configuration files.
     - Makefile: Automates tasks related to building, testing, and deployment.
     - imgs/: Stores images used in the project, such as visuals for reports or documentation.
     - notebooks/: Contains Jupyter notebooks used for exploratory data analysis (EDA) and experimentation, such as 02-capstone-project.ipynb.
     - requirements/: Includes files that specify project dependencies:
     - Pipfile: Used for dependency management with Pipenv.
     - requirements.txt: Lists all Python libraries needed to run the project.
     - src/: Main source code directory, divided into the following subdirectories:
     - inference/: Code for making predictions with the trained model (e.g., predict.py).
     - tests/: Contains testing scripts to ensure code reliability (e.g., test.py).
     - training/: Scripts related to model training and pipeline creation, such as train.py and pipeline.joblib.
     - .dockerignore: Specifies files and directories to ignore during the Docker build process.
     - README.md: The main documentation file for the project, providing an overview, setup instructions, and usage guidelines.

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

**Examples:**

![image](https://github.com/user-attachments/assets/59f3c5fd-1a03-46cb-ab39-e75dbbe853a6)

![image](https://github.com/user-attachments/assets/5f03288c-b097-453b-bcfa-0c8811fd7960)

![image](https://github.com/user-attachments/assets/93e6e933-c7ac-44ae-96ac-da4234d9071a)

### Visualizations

![image](https://github.com/user-attachments/assets/29586746-84ee-41ad-96d1-aac373180559)

![image](https://github.com/user-attachments/assets/d7ff4680-0de7-4b7c-9723-ed8b32c00956)

![image](https://github.com/user-attachments/assets/d9c629b7-8557-4966-810f-e1376a80a564)

![image](https://github.com/user-attachments/assets/784c04e4-01d7-4e55-9bca-b29bf46c4b8b)


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

![image](https://github.com/user-attachments/assets/2b561ca7-6d21-422f-8177-7a412341f4b5)

**Response:**

![image](https://github.com/user-attachments/assets/01e1c756-d6f1-4080-a0d0-5015c7ba8765)


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
