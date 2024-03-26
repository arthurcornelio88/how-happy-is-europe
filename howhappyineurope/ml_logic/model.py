import numpy as np
import xgboost as xgb

def initialize_model():
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

    model.fit(
        X_train,
        y_train
    )
    print(f"✅ Model trained")

    return model
