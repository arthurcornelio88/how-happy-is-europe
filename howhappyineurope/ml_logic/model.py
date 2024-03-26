import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.linear_model import LinearRegression
from colorama import Fore, Style

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
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    model.fit(
        X_train,
        y_train
    )
    print(f"✅ Model trained")
    #print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model

def evaluate_model(
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X_test)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    # accuracy for 3 classes of happiness
    accuracy_score(res_int_df["test_reduced"], res_int_df["pred_reduced"]) * 100

    # metrics = model.evaluate(
    #     x=X,
    #     y=y,
    #     batch_size=batch_size,
    #     verbose=0,
    #     # callbacks=None,
    #     return_dict=True
    # )

    # loss = metrics["loss"]
    # mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
