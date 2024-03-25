import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.linear_model import LinearRegression
from colorama import Fore, Style

def initialize_model(input_shape: tuple):
    """
    Initialize the Linear regression model
    """

    model = xgb.XGBRegressor(
    booster="gbtree",
    n_jobs=4,
    random_state=42,
    verbosity=0,)

    print("✅ Model initialized")

    return model

def train_model(
        model,
        X_train: np.ndarray,
        y_train: np.ndarray):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    model.fit(
        X_train,
        y_train
    )
    print(f"✅ Model trained")
    #print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model
