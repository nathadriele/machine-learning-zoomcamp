## Ford Car Price Prediction

![image](https://github.com/user-attachments/assets/8afdf9f6-d86e-4040-92bf-ce42fc0233d3)

### Problem Description

Determining the right selling price is vital for used car sellers—be it dealerships or individual owners. Pricing too low leads to lost profits, while pricing too high deters potential buyers. This project uses Machine Learning to predict used Ford car prices based on attributes like year, mileage, fuelType, and more. The project also explores a binary classification approach (e.g., “Is the price above/below a certain threshold?”) to show additional metrics and insights.

Ford Car Price Prediction leverages Machine Learning techniques to accurately predict used Ford car prices based on various attributes. The objective is to:

- Prevent underpricing, leading to profit losses.
- Avoid overpricing, leading to buyer deterrence.
- Foster confidence in fair-market deals by providing data-driven valuations.
- Through data exploration, modeling, and a Flask API deployment, this project delivers a scalable solution for accurate used-car price predictions.

### Objectives

1. Data Preparation and Cleaning
- Remove outliers, handle missing values, ensure consistent data types.
2. Feature Engineering
- Categorize or transform attributes (e.g., year buckets).
3. Model Training
- Test multiple regression algorithms (Linear, Ridge, Lasso, XGBoost) for price.
4. Model Evaluation
- Use RMSE, R² (for regression) and potentially Precision, Recall, F1 (for classification).
5. Deployment
- Provide a production-ready model via Flask (Docker/Kubernetes).

### Table of Contents

- [1. Dataset](#dataset)
   - [1.1 Description](#description)
   - [1.2 Sources](#sources)
- [2. Requirements](#requirements)
- [3. Installation](#installation)
- [4. Exploratory Data Analysis (EDA)](#exploratory)
- [5. Data Preparation](#data)
- [6. Model Training and Evaluation](#model)
   - [6.1 Regression Models](#regression)
   - [6.2 Classification Models](#classification)
   - [6.3 Evaluation Metrics](#evaluation)
   - [6.4 Hyperparameter Tuning](#hyperparameter)
- [7. Save and Load the Model](#save)
- [8. Model Deployment](#deployment)
   - [8.1 Flask Web Service](#flask)
   - [8.2 Dockerization and Kubernetes](#dockerization)
- [9. Testing](#testing)
- [10. API Usage (Example)](#api)
- [11. Conclusion](#conclusion)
- [12. Contribution](#contribution)

### Dataset

![image](https://github.com/user-attachments/assets/b2174753-4c56-4096-b1e5-b90b39b0a9e9)

File: ford_car_price_prediction.csv

- Columns (9 total):
   - model (Fiesta, Focus, Kuga, etc.)
   - year (range 2013–2020)
   - price (the regression target)
   - transmission (Manual, Automatic, Semi-Auto, etc.)
   - mileage (distance driven)
   - fuelType (Petrol, Diesel, …)
   - tax (annual road tax in £)
   - mpg (miles per gallon)
   - engineSize (liters)






