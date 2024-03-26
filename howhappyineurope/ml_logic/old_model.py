import numpy as np
import pandas as pd

# import linear regression model
from sklearn.linear_model import LinearRegression

from colorama import Fore, Style

def initialize_model(input_shape: tuple):
    """
    Initialize the Linear regression model
    """

    model = LinearRegression()

    print("✅ Model initialized")

    return model

def train_model(
        model,
        X: np.ndarray,
        y: np.ndarray):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    model.fit(
        X,
        y
    )
    print(f"✅ Model trained")
    #print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model
