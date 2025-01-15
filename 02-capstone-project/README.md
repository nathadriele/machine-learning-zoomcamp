## Uber Ride Price Prediction

### Table of Contents
- Overview
- Features
- Technologies Used
- Installation
- Usage
   - Data Preprocessing and Model Training
   - Model Deployment via API
- Project Structure
- Data Description
- Model Evaluation
- Contributing
- License
- Contact

### Overview

The Uber Ride Price Prediction project aims to develop a machine learning model that accurately predicts the price of Uber rides based on various features such as ride location, time, weather conditions, and ride specifics. This project encompasses data preprocessing, feature engineering, model training, hyperparameter tuning, and deployment through a RESTful API.

### Features

- **Data Cleaning & Preprocessing:** Handling missing values, encoding categorical variables, and scaling numerical features.
- **Feature Selection:** Utilizing Recursive Feature Elimination (RFE) to identify the most impactful features.
- **Model Training & Evaluation:** Training multiple regression models and evaluating their performance using metrics like RMSE, MAE, and R².
- **Hyperparameter Tuning:** Optimizing model performance using GridSearchCV.
- **Model Deployment:** Providing a RESTful API for real-time price predictions using Flask.
- **Pipeline Automation:** Implementing a structured pipeline to ensure consistent data transformations and model training.

### Technologies Used

- Programming Language: Python 3.x
- Libraries:
   - Data Manipulation: pandas, numpy
   - Machine Learning: scikit-learn, xgboost
   - Model Serialization: joblib
   - API Deployment: Flask
- Tools:
   - Development Environment: Jupyter Notebook, VS Code
   - Version Control: Git & GitHub

### Installation

#### Prerequisites

- Python 3.7 or higher
- Git

### Steps

1. Clone the Repository:

```
git clone https://github.com/nathadriele/uber-price-prediction.git
cd uber-price-prediction
```

2. Create a Virtual Environment:

```python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies:

```
pip install -r requirements.txt
```


