import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import xgboost as xgb

RANDOM_SEED = 42

INPUT_FILE = 'inputs/rideshare_kaggle.csv'
OUTPUT_FILE = 'price_prediction.bin'

DROP_COLS = [
    'timezone', 'product_id', 'short_summary', 'long_summary', 'windGustTime', 
    'temperatureHigh', 'temperatureHighTime', 'temperatureLow', 'temperatureLowTime',
    'apparentTemperatureHigh', 'apparentTemperatureHighTime', 'apparentTemperatureLow', 
    'apparentTemperatureLowTime', 'icon', 'dewPoint', 'pressure', 'windBearing', 
    'cloudCover', 'uvIndex', 'visibility.1', 'ozone', 'sunriseTime', 'sunsetTime', 
    'moonPhase', 'precipIntensityMax', 'uvIndexTime', 'temperatureMin', 
    'temperatureMinTime', 'temperatureMax', 'temperatureMaxTime', 
    'apparentTemperatureMin', 'apparentTemperatureMinTime', 'apparentTemperatureMax', 
    'apparentTemperatureMaxTime'
]

TRAIN_COLS = [
    'distance', 'surge_multiplier', 'latitude', 'longitude', 'temperature', 
    'apparenttemperature', 'precipintensity', 'precipprobability', 'humidity', 
    'windspeed', 'windgust', 'visibility', 'source', 'name', 'hour', 'day'
]

def prepare_df(df_train, df_val, cols):
    """
    Prepare training and validation datasets by converting them to dictionaries
    and applying DictVectorizer.
    """
    dv = DictVectorizer(sparse=False)
    
    df_train = df_train[cols]
    df_val = df_val[cols]

    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    return dv, X_train, X_val

def train_model(X_train, y_train):
    """
    Train an XGBoost regressor with specified hyperparameters.
    """
    model = xgb.XGBRegressor(
        colsample_bytree=1.0,
        learning_rate=0.2,
        max_depth=7,
        n_estimators=200,
        subsample=1.0,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    return model

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        sys.exit(1)

    print("Reading data...")
    df = pd.read_csv(INPUT_FILE)
    df2 = df.drop(DROP_COLS, axis=1)
    df2.columns = df2.columns.str.replace(' ', '_').str.lower()
    df2 = df2.set_index('id')

    print("Splitting data...")
    df_full_train, df_test = train_test_split(df2, test_size=0.2, random_state=RANDOM_SEED)

    print("Handling missing values...")
    df_full_train = df_full_train.dropna()
    y_full_train = df_full_train.price.values
    y_test = df_test.price.values
    df_full_train = df_full_train.drop(['price'], axis=1)
    df_test = df_test.drop(['price'], axis=1)

    print("Preparing data for training...")
    dv, X_full_train, X_test = prepare_df(df_full_train, df_test, TRAIN_COLS)

    print("Training the model...")
    model = train_model(X_full_train, y_full_train)

    print("Evaluating the model...")
    valid_indices = ~np.isnan(y_test)
    X_test_notna = X_test[valid_indices]
    y_test_notna = y_test[valid_indices]
    gbm_pred = model.predict(X_test_notna)
    rmse_test = mean_squared_error(y_test_notna, gbm_pred, squared=False)
    print(f"XGBoost on the test set gives RMSE: {rmse_test:.2f}")

    print(f"Saving the model to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump((dv, model), f_out)

    print("Model saved successfully.")

if __name__ == "__main__":
    main()
