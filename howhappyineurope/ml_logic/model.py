# TODO : need to fix it
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
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

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, minmax_Y):
    """
    Evaluate trained model performance on the dataset.

    :param model: Trained model to evaluate.
    :param X_test: Test features.
    :param y_test: True labels for test data.
    :param minmax_Y: Scaler object used for inverse transformation.
    """
    if model is None:
        print("\n❌ No model to evaluate")
        return None

    print(f"\nEvaluating model on {len(X_test)} rows...")

    # Predict the labels of the test set
    y_pred = model.predict(X_test)

    # Calculate the Mean Absolute Error (MAE) of the predictions
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae}")

    # Assuming y_pred and y_test need to be inversely transformed for accuracy calculation
    yy_pred = np.round(minmax_Y.inverse_transform(y_pred[:, np.newaxis]))
    yy_test = minmax_Y.inverse_transform(y_test[:, np.newaxis])  # Ensure y_test is correctly shaped

    # Calculate the accuracy of the predictions (for categorized outputs)
    accuracy = accuracy_score(yy_test, yy_pred) * 100
    print(f"Accuracy: {accuracy}%")

    # Return both metrics for further use
    return {"mae": mae, "accuracy": accuracy}

def save_model(model):
    #Save model locally
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

def load_model():
    # Load Pipeline from pickle file
    model = pickle.load(open("models/bh_model.pkl","rb"))

    return model
