import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import logging
import warnings
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", category=FutureWarning)

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logging.info(f'Dataset loaded successfully from {filepath}')
        return df
    except FileNotFoundError:
        logging.error(f'File {filepath} not found.')
        raise
    except Exception as e:
        logging.error(f'Error loading dataset: {e}')
        raise

data = './hypertension_dataset.csv'
df = load_data(data)

target_label = 'Outcome'

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

columns_to_clean = ['Cholesterol', 'BloodPressure', 'PhysicalActivity', 'SodiumIntake', 'BMI', 'Age']
for column in columns_to_clean:
    df = remove_outliers_iqr(df, column)
    logging.info(f'Outliers removed from column: {column}')

df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, np.inf],
                            labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

data = pd.get_dummies(df, columns=['BMI_Category'], drop_first=True)

# Split dataset to train and test
categorical = ['BMI_Category_Normal', 'BMI_Category_Overweight', 'BMI_Category_Obese']
numerical = [col for col in data.columns if col not in categorical + [target_label]]

df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=1)
df_train = df_full_train[categorical + numerical]
df_val = df_test[categorical + numerical]
y_train = df_full_train[target_label].values
y_val = df_test[target_label].values

# Vectorize features
dv = DictVectorizer(sparse=False)
train_dicts = df_train.fillna(0).to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)

# Model Training
model_lr = LogisticRegression(solver='liblinear', random_state=42)

param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Penalty type (L1 or L2 regularization)
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    estimator=model_lr,
    param_grid=param_grid_lr,
    cv=5,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)

try:
    logging.info('Starting model training with GridSearchCV...')
    grid_search.fit(X_train, y_train)
    logging.info('Model training completed.')
except Exception as e:
    logging.error(f'Error during model training: {e}')
    raise

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]

logging.info(f"Best Hyperparameters: {best_params}")
print("Classification Report:")
print(classification_report(y_val, y_pred))
print(f"Precision: {precision_score(y_val, y_pred)}")
print(f"Recall: {recall_score(y_val, y_pred)}")
print(f"F1 Score: {f1_score(y_val, y_pred)}")
print(f"AUC-ROC Score: {roc_auc_score(y_val, y_pred_proba)}")

# Save the model
def save_model(dv, model):
    output_file = 'model.bin'
    try:
        with open(output_file, 'wb') as f_out:
            pickle.dump((dv, model), f_out)
        logging.info(f'Model saved successfully to {output_file}')
    except Exception as e:
        logging.error(f'Error saving the model: {e}')
        raise
    return output_file

output_file = save_model(dv, best_model)
print(f'The model is saved to {output_file}')