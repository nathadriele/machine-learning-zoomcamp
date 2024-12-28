## Ford Car Price Prediction 

![image](https://github.com/user-attachments/assets/8afdf9f6-d86e-4040-92bf-ce42fc0233d3)

Used Car Valuation with Machine Learning for Informed Pricing Decisions. 

### Problem Description

Determining the right selling price is vital for used car sellers—be it dealerships or individual owners. Pricing too low leads to lost profits, while pricing too high deters potential buyers. This project uses Machine Learning to predict used Ford car prices based on attributes like year, mileage, fuelType, and more. The project also explores a binary classification approach (e.g., “Is the price above/below a certain threshold?”) to show additional metrics and insights. 🚗💸📈

### Objectives

#### 1. Data Preparation and Cleaning
- Remove outliers, handle missing values, ensure consistent data types.

#### 2. Feature Engineering
- Categorize or transform attributes (e.g., year buckets).

#### 3. Model Training
- Test multiple regression algorithms (Linear, Ridge, Lasso, XGBoost) for price.

#### 4. Model Evaluation
- Use RMSE, R² (for regression) and potentially Precision, Recall, F1 (for classification).

#### 5. Deployment
- Provide a production-ready model via Flask (Docker/Kubernetes).

### Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Dataset](#2-dataset)
- [3. Requirements](#3-requirements)
- [4. Installation](#4-installation)
- [5. Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
- [6. Data Preparation](#6-data-preparation)
- [7. Model Training and Evaluation](#7-model-training-and-evaluation)
    - [7.1 Regression Models](#71-regression-models)
    - [7.2 Classification Models](#72-classification-models)
    - [7.3 Evaluation Metrics](#73-evaluation-metrics)
    - [7.4 Hyperparameter Tuning](#74-hyperparameter-tuning)
- [8. Model Persistence (Saving and Loading)](#8-model-persistence-saving-and-loading)
- [9. Model Deployment](#9-model-deployment)
    - [9.1 Flask Web Service](#91-flask-web-service)
    - [9.2 Dockerization and Kubernetes](#92-dockerization-and-kubernetes)
- [10. Testing](#10-testing)
- [11. API Usage Example](#11-api-usage-example)
- [12. Model Evaluation Visualizations](#12-model-evaluation-visualizations)
- [13. Conclusion](#13-conclusion)
- [14. Contribution](#14-contribution)


### 1. Project Overview

The Ford Car Price Prediction project aims to accurately predict the selling price of used Ford vehicles. Precise pricing ensures:

- Fair Market Value: Sellers avoid under- or overpricing.
- Higher Profit Margins: Data-driven decisions increase trust and reduce guesswork.
- Customer Satisfaction: Competitive, realistic pricing fosters better buyer engagement.

This repository showcases:

- Data Collection and Cleaning
- Exploratory Data Analysis (EDA)
- Regression approaches (and optionally Classification for threshold-based insights)
- Model Training & Hyperparameter Tuning
- Model Evaluation (RMSE, R², Precision, Recall, etc.)
- Deployment with Flask (plus Docker/Kubernetes examples)

### 2. Dataset

![image](https://github.com/user-attachments/assets/b2174753-4c56-4096-b1e5-b90b39b0a9e9)

File: ford_car_price_prediction.csv

- Source: Adapted dataset from Kaggle or local sources (file: ford_car_price_prediction.csv).
- Size: ~1,460 entries (reduced to ~1,358 after outlier removal).
- Features (9 columns):
1. model (categorical)
2. year (integer)
3. price (integer, target for regression)
4. transmission (categorical)
5. mileage (integer)
6. fuelType (categorical)
7. tax (integer)
8. mpg (float)
9. engineSize (float)
- Target: price (used in regression).

### 3. Requirements

- Python: 3.9+
- Libraries (main):
   - pandas, numpy, scikit-learn, xgboost
   - matplotlib, seaborn (for EDA/visuals)
   - joblib, pickle (for model serialization)
   - flask (for API), streamlit (optional UI)
   - docker, kubernetes (optional container orchestration)
A requirements.txt/Pipfile is provided to facilitate quick installation.

### 4. Installation

- Clone the repo:

`git clone https://github.com/nathadriele/ford-car-price-prediction.git`

`cd ford-car-price-prediction`

- Create and activate a virtual environment:

`python -m venv venv`

`source venv/bin/activate  # On Windows: venv\Scripts\activate`

- Install dependencies:

`pip install -r requirements.txt`

or

`pipenv install --deploy --system`

### 5. Exploratory Data Analysis (EDA)

We conducted an extensive EDA to understand patterns, distributions, and potential anomalies:

#### 1. Descriptive Stats (df.describe(), df.info()):

- price ranges ~3,691 to ~42,489 pre-outlier removal.
- year from 2013 to 2020.
- mileage wide range (up to ~88,927 before cleaning).

#### 2. Missing Values and Before and After Outlier Removal:

![image](https://github.com/user-attachments/assets/284f5605-3ef9-4a50-b0a8-afa12b2e2737)

![image](https://github.com/user-attachments/assets/7584a9d1-ad5a-49b0-a1b1-a2e105e44a04)

#### 3. Visualizations:

- Histograms & Boxplots: Detected outliers in price, mileage, mpg, and tax.

![image](https://github.com/user-attachments/assets/43b31e0f-c9ec-46b5-8e79-395e245a5ff3)

- Scatter Plots: mileage inversely correlates with price; year positively correlates.

![image](https://github.com/user-attachments/assets/a53513df-edb4-4d76-a3c4-34b8aa59ee07)

- Correlation Heatmap: year has moderate correlation (0.63) to price; mileage negative (-0.46).

![image](https://github.com/user-attachments/assets/7038651c-4a48-4ae5-a223-f823403e19be)

- Price Distribution by Model.

![image](https://github.com/user-attachments/assets/a78a0ebe-5d07-4732-927f-91830840d0cc)

- Engine Size vs Price with Regression Line.

![image](https://github.com/user-attachments/assets/f1dd7847-e17b-4315-b21d-ee0ce1fe9376)

- Distribution of average price by year.

![image](https://github.com/user-attachments/assets/494ccbca-30da-44c7-bf3f-60455b2ff716)

### 6. Data Preparation

#### 1. Outlier Removal

- Used the IQR method (1.5 × IQR) on price, mileage, tax, mpg, engineSize.
- Reduced dataset from 1,460 to ~1,358 rows.

#### 2. Feature Engineering

- Potential binning of mpg or engineSize.
- Adjusted or standardized numeric columns for certain models.

#### 3. Train-Validation Split

- `80/20 split:` train_test_split(df, test_size=0.2, random_state=42).

### 7. Model Training and Evaluation

### 7.1 Regression Models

Goal: Predict price (numeric).

- `Linear Regression`
- `Ridge Regression`
- `Lasso Regression`
- `XGBoost Regressor`

Approx. Results:

| Model         | RMSE | R² |
|---------------|-----------|--------|
| Linear Regression |  1448.85 | 0.8606  | 
| Ridge Regression  |  1448.76 |  0.8606 |
| Lasso Regression  |  1448.85 |  0.8606 |
| XGBoost |  1059.46 | 0.9255  |  

### 7.2 Classification Models

Goal: Classify cars as high-priced vs. low-priced.
(For demonstration, threshold = £12,000.)

1. `Logistic Regression`
2. `Random Forest`
3. `XGBoost Classifier`

Approx. Classification Results:

**Logistic Regression**

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.75      | 0.73   | 0.74     | 420    |
| **1**         | 0.80      | 0.82   | 0.81     | 560    |
| **Accuracy**  |           |        | 0.78     | 980    |
| **Macro Avg** | 0.78      | 0.77   | 0.77     | 979    |
| **Weighted Avg** | 0.78   | 0.78   | 0.78     | 980    |
| **AUC-ROC** | ~0.82   |     |      |      |

**Random Forest**

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.79      | 0.75   | 0.74     | 450    |
| **1**         | 0.83      | 0.86   | 0.81     | 568    |
| **Accuracy**  |           |        | 0.78     | 980    |
| **Macro Avg** | 0.81      | 0.81   | 0.77     | 979    |
| **Weighted Avg** | 0.82   | 0.82   | 0.78     | 980    |
| **AUC-ROC** | ~0.87   |     |      |      |

**XGBoost Classifier**

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.82      | 0.76   | 0.79     | 446    |
| **1**         | 0.85      | 0.88   | 0.86     | 564    |
| **Accuracy**  |           |        | 0.83     | 982    |
| **Macro Avg** | 0.84      | 0.83   | 0.81     | 978    |
| **Weighted Avg** | 0.84   | 0.83   | 0.79     | 972    |
| **AUC-ROC** | ~0.89   |     |      |      |


### 7.3 Evaluation Metrics

**For Regression**

- `RMSE (Root Mean Squared Error):` Lower is better.
- `R²:` Measures how much variance is explained (closer to 1 is better).

**For Classification**

- `Precision, Recall, F1:` from confusion matrix.
- `AUC-ROC:` area under the ROC curve; robust metric for classification thresholds.

### 7.4 Hyperparameter Tuning

Key parameters:

- Ridge/Lasso: alpha grid from 0.001 to 100.
- XGBoost: {n_estimators, max_depth, learning_rate, …}
- RandomForest: {n_estimators, max_depth, min_samples_split, …}

Sample best params for `XGBoost Regressor`:

![image](https://github.com/user-attachments/assets/5d843167-8d8c-4ef3-b1c1-057a0eb1353b)

### 8. Model Persistence (Saving and Loading)

![image](https://github.com/user-attachments/assets/953a30b1-1f5f-4c03-8765-9810c4e19a93)

### 9. Model Deployment

`app.py:` 

```rb
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
pipeline = joblib.load("pipeline.joblib") 

@app.route("/")
def index():
    return "Ford Car Price Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df_input = pd.DataFrame([data])
    prediction = pipeline.predict(df_input)[0]
    return jsonify({"predicted_price": float(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

### 9.1 Flask Web Service

After identifying XGBoost as the best-performing model, the next step is to make these predictions accessible to external systems or users via a Flask web service. Below is a simplified example of how to deploy the trained model as an API.

### 9.2 Dockerization and Kubernetes

`Dockerfile:`

![image](https://github.com/user-attachments/assets/dae9a0f2-cce6-46f3-8190-87b8087093e3)

**Build & Run** locally:

![image](https://github.com/user-attachments/assets/3de770e5-4495-4a8b-bb57-2755c125ca99)

**Kubernetes:** Use provided deployment.yaml & service.yaml for cluster-level orchestration.

### 10. Testing

Three main scenarios:

#### 1. Local:

- python app.py
- curl -X POST http://localhost:5000/predict ...

#### 2. Docker:

- docker run -p 5000:5000 ford-price-prediction:latest

#### 3. test.py script:

![image](https://github.com/user-attachments/assets/c6927816-4939-440d-8fd3-00ab4ce3290b)

### 11. API Usage Example

**POST** request:

![image](https://github.com/user-attachments/assets/139a70cb-2e0a-4ddc-b02c-a62509f6af49)

Example response:

![image](https://github.com/user-attachments/assets/1cf04905-8f07-4a34-9dc6-466f3c26602b)

### 12. Model Evaluation Visualizations

![image](https://github.com/user-attachments/assets/717e5450-a7a2-44b2-93a4-cfdde42f2848)

- Closeness to the Ideal line: Points near this diagonal suggest accurate predictions.
- Overestimation / Underestimation: Points significantly above the diagonal are underpredicted (the model’s estimates are too low), while points below the diagonal are overpredicted (the model’s estimates are too high).
- In the example image for XGBoost, you can see that most points are well-aligned with the line, indicating a strong relationship between the car attributes and the predicted price.

![image](https://github.com/user-attachments/assets/b55a835f-dc1e-4e6f-b4c1-d302aa0113e4)


- Center around zero: A concentration of residuals around zero suggests that, on average, the model is neither systematically overpredicting nor underpredicting.
- Skewness or outliers: Heavy tails (extreme positive or negative residuals) indicate outliers or instances where the model struggles.
- From the XGBoost histogram example, the bulk of residuals lie in a range centered near zero, reflecting accurate predictions, though there is a slight tail extending to the right, meaning some prices are overestimated.

### 13. Conclusion

This Ford Car Price Prediction project is part of a Machine Learning pipeline practice from the Zoomcamp course, illustrating real-world steps—from EDA to production—in delivering scalable and accurate predictions. Key highlights:

- Detailed EDA & Outlier Removal: Ensured cleaner data for robust modeling.
- Multiple Regression Models: XGBoost excelled with an RMSE of ~1059 and R² of ~0.925.
- Classification: Showed how threshold-based predictions might categorize “high” vs. “low” price vehicles with strong AUC-ROC (~0.90).
- Hyperparameter Tuning: Employed GridSearchCV for optimized parameters.
- Deployment: Demonstrated a Flask-based API, containerized with Docker, and orchestrated via Kubernetes.
In future work, additional features (e.g., geographic data, seller type) or advanced ensemble techniques may push performance further. The Zoomcamp approach ensures easily replicable workflows, reinforcing best practices in ML development and MLOps.

### 14. Contribution

Contributions are welcome!

Thank you for reading and using the Ford Car Price Prediction repository, my friend!



