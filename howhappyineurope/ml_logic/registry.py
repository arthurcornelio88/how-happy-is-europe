import os
import time
import pickle

from colorama import Fore, Style

from howhappyineurope.params import *

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")

def save_model(model):
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """

    # Save model locally
    model_path = os.path.join("models", f"model.pkl")
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    final_path = os.path.join(dir_path, model_path)
    print(final_path)

    with open(final_path, "wb") as file:
        pickle.dump(model, file)

    print("✅ Model saved locally")

    return None

def load_model():
    """
    Return a saved model:
    - locally (latest one in alphabetical order)

    """
    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    model_path = os.path.join("models", f"model.pkl")
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    local_model_directory = os.path.join(dir_path, model_path)
    print(model_path)
    print(dir_path)
    print(local_model_directory)

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    latest_model = pickle.load(open(local_model_directory,"rb"))

    print("✅ Model loaded from local disk")

    return latest_model
