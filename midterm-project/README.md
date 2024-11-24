# Hypertension Prediction

## Problem Description

Hypertension, commonly referred to as high blood pressure, is a pervasive and silent health condition that affects millions of individuals worldwide. Characterized by the persistent elevation of blood pressure in the arteries, hypertension serves as a significant risk factor for a multitude of severe health issues, including heart diseases, strokes, kidney failure, and other vascular complications. According to the World Health Organization (WHO), hypertension is responsible for approximately 7.5 million deaths each year, making it one of the leading causes of mortality globally.

Despite its widespread prevalence, hypertension often remains undetected in its early stages due to its asymptomatic nature. Many individuals are unaware of their high blood pressure levels until they experience a critical health event, such as a heart attack or stroke. This delayed diagnosis hampers the effectiveness of preventive measures and increases the burden on healthcare systems.

Early detection of individuals at high risk of developing hypertension is crucial for implementing timely and effective preventive interventions. Identifying these individuals allows for lifestyle modifications, medical treatments, and regular monitoring, which can significantly reduce the risk of severe complications and improve overall quality of life.

This project addresses the critical need for early hypertension detection by leveraging advanced Machine Learning (ML) techniques to develop predictive classification models. By analyzing various health parameters and demographic factors, the models aim to accurately predict the risk of hypertension in individuals. This proactive approach facilitates targeted interventions, enabling healthcare providers to prioritize resources and tailor preventive strategies to those most in need.

The dataset utilized in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/prosperchuks/health-dataset?select=hypertension_data.csv) and specifically comprises the `hypertension_data.csv` file (1.09 MB). This dataset encompasses detailed information about individuals, including demographic characteristics, clinical measurements, and other health-related attributes relevant to hypertension detection. With 26,083 records and 14 variables, the dataset provides a robust foundation for training and evaluating Machine Learning models aimed at predicting hypertension risk.

By developing accurate and reliable predictive models, this project aims to empower individuals and healthcare professionals with actionable insights, ultimately contributing to the reduction of hypertension-related health risks and associated healthcare costs. The comprehensive approach, encompassing data preparation, model training, evaluation, and deployment, ensures that the developed solution is both effective and scalable for real-world applications.

## Objectives

1. **Data Preparation and Cleaning:** Ensure data quality through handling missing values, removing outliers, and transforming variables.
2. **Feature Engineering:** Create and select features that significantly contribute to hypertension prediction.
3. **Development of Predictive Models:** Train and optimize various Machine Learning models to identify the most suitable for the problem.
4. **Evaluation and Selection of the Best Model:** Use performance metrics and cross-validation to select the most effective model.
5. **Model Deployment:** Make the trained model available through an API for real-world application.
6. **Documentation and MLOps:** Implement MLOps practices to ensure reproducibility, scalability, and continuous maintenance of the model.

## Index

### Table of Contents

- [Dataset](#dataset)
  - [Description](#description)
  - [Sources](#sources)
- [Requirements](#requirements)
- [Installation](#installation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Univariate Analysis](#univariate-analysis)
  - [Bivariate Analysis](#bivariate-analysis)
  - [Outlier Detection](#outlier-detection)
  - [Correlation Heatmap](#correlation-heatmap)
- [Data Preparation](#data-preparation)
  - [Handling Missing Values](#handling-missing-values)
  - [Removing Outliers](#removing-outliers)
  - [Feature Scaling](#feature-scaling)
- [Feature Engineering](#feature-engineering)
  - [Creating Categories](#creating-categories)
  - [Feature Importance Analysis](#feature-importance-analysis)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [Models Used](#models-used)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Model Selection](#model-selection)
- [Model Selection and Final Evaluation](#model-selection-and-final-evaluation)
- [Threshold Analysis for Model Performance](#threshold-analysis-for-model-performance)
- [Cross-Validation Report](#cross-validation-report)
- [Save and Load the Model](#save-and-load-the-model)
- [Model Deployment](#model-deployment)
  - [Flask Web Service](#flask-web-service)
  - [Dockerization](#dockerization)
- [Testing](#testing)
  - [Test Scenario 1: Local Service](#test-scenario-1-local-service)
  - [Test Scenario 2: Dockerized Service](#test-scenario-2-dockerized-service)
- [API Usage Example](#api-usage-example)
- [Conclusion](#conclusion)
- [Contribution](#contribution)

## Dataset

### Description

The dataset used in this project contains detailed information about individuals, including demographic and health parameters. Each record describes characteristics that may influence the risk of hypertension.

- **Total Records:** 26,083
- **File Size:** 1.09 MB
- **Target Variable:** `target` (1 for hypertensive, 0 for non-hypertensive)
- **Descriptive Columns:**

| Column    | Description                                                                                      |
|-----------|--------------------------------------------------------------------------------------------------|
| age       | Age of the individual (years)                                                                    |
| sex       | Sex of the individual (0 = female, 1 = male)                                                     |
| cp        | Chest pain type (categories 0 to 3)                                                              |
| trestbps  | Resting blood pressure (mm Hg)                                                                    |
| chol      | Serum cholesterol (mg/dl)                                                                         |
| fbs       | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)                                           |
| restecg   | Resting electrocardiographic results (categories 0 to 2)                                         |
| thalach   | Maximum heart rate achieved during exercise                                                       |
| exang     | Exercise-induced angina (1 = yes, 0 = no)                                                        |
| oldpeak   | ST depression induced by exercise relative to rest                                                  |
| slope     | Slope of the peak exercise ST segment (categories 0 to 2)                                        |
| ca        | Number of major vessels colored by fluoroscopy (0 to 4)                                          |
| thal      | Fixed defect, reversible defect, or normal                                                             |
| target    | Target variable indicating the presence of hypertension (1) or absence (0)                         |

### Sources

This dataset was obtained from the [Kaggle](https://www.kaggle.com/datasets/prosperchuks/health-dataset?select=hypertension_data.csv) platform and is widely used in Machine Learning research focused on health diagnostics and predictive analyses. The original data source has been adapted for this project, ensuring privacy and compliance with ethical guidelines for using health data.

## Requirements

- **Python 3.9+**
- **Libraries:**
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - flask
  - pickle
  - matplotlib
  - seaborn
  - gunicorn
  - tqdm

## Installation

1. **Clone the Repository:**
    ```bash
    git clone git@github.com:nathadriele/machine-learning-zoomcamp.git
    cd midterm-project
    ```

2. **Create and Activate a Virtual Environment:** 
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Exploratory Data Analysis (EDA) 

⚠️ **With some print examples taken from the project notebook. For all the details, access the notebook.**

### Univariate Analysis

We analyzed the individual distribution of each numerical and categorical variable to better understand the nature of the data.

#### Age Distribution

![alt text](images/image.png)

#### Cholesterol Distribution

![alt text](images/image-1.png)

### Bivariate Analysis

![alt text](images/image-2.png)

We explored the relationships between independent variables and the target variable (`target`).

#### Boxplots of Numerical Variables by Target

![alt text](images/image-3.png)

#### Count Plots of Categorical Variables by Target

![alt text](images/image-4.png)

#### Plot the Distribution of Age Categories in Relation to Target

![alt text](images/image-5.png)

### Plot of distributions for numerical columns and frequency

![alt text](images/image-6.png)

### Outlier Detection

We used boxplots to identify and remove outliers in numerical variables.

![alt text](images/image-7.png)

### Correlation Heatmap

We analyzed the correlations between numerical variables to identify potential multicollinearities.

![alt text](images/image-8.png)

## Data Preparation

### Handling Missing Values

We identified and handled missing values in relevant columns. For example, we replaced missing values in columns like `sex` with the median or another appropriate strategy.

### Removing Outliers

We applied the Interquartile Range (IQR) method to remove outliers from columns such as `age`, `trestbps`, `chol`, `thalach`, and `oldpeak`.

### Feature Scaling

We normalized or standardized numerical features to ensure that all variables contribute equally to the model.

## Feature Engineering

### Creating Categories

We created new categorical features, such as `Age_Category`, based on age ranges, to capture non-linear patterns related to hypertension risk.

### Feature Importance Analysis

We used feature selection techniques and correlation analyses to identify the most relevant variables for hypertension prediction.

## Model Training and Evaluation

### Models Used

We trained various classification models, including:

- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Ensemble Voting Classifier**

### Hyperparameter Tuning

We utilized `GridSearchCV` and `RandomizedSearchCV` to optimize the hyperparameters of each model, aiming to improve their performance.

#### Logistic Regression 

# LogisticRegression Classification Report

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.72      | 0.60   | 0.65     | 2153    |
| **1**         | 0.73      | 0.80   | 0.76     | 2737    |
| **Accuracy**  |           |        | 0.72     | 4890    |
| **Macro Avg** | 0.73      | 0.70   | 0.70     | 4890    |
| **Weighted Avg** | 0.73   | 0.72   | 0.72     | 4890    |

**AUC-ROC Score**: 0.78

#### Random Forest Classifier

# RandomForest Classification Report

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.80      | 0.65   | 0.72     | 2153    |
| **1**         | 0.76      | 0.85   | 0.80     | 2737    |
| **Accuracy**  |           |        | 0.75     | 4890    |
| **Macro Avg** | 0.78      | 0.75   | 0.76     | 4890    |
| **Weighted Avg** | 0.78   | 0.75   | 0.76     | 4890    |

**AUC-ROC Score**: 0.85

#### XGBoost Classifier

# XGBoost Classification Report

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.88      | 0.75   | 0.81     | 2153    |
| **1**         | 0.80      | 0.88   | 0.84     | 2737    |
| **Accuracy**  |           |        | 0.80     | 4890    |
| **Macro Avg** | 0.84      | 0.81   | 0.82     | 4890    |
| **Weighted Avg** | 0.84   | 0.80   | 0.81     | 4890    |

**AUC-ROC Score**: 0.89

#### Ensemble Voting Classifier

# Ensemble Learning - Voting Classifier Report

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.82      | 0.70   | 0.75     | 2153    |
| **1**         | 0.76      | 0.85   | 0.80     | 2737    |
| **Accuracy**  |           |        | 0.77     | 4890    |
| **Macro Avg** | 0.79      | 0.78   | 0.77     | 4890    |
| **Weighted Avg** | 0.79   | 0.77   | 0.77     | 4890    |

**AUC-ROC Score**: 0.85



