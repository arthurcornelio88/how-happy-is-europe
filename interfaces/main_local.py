from ml_logic.preprocessor import preprocess_data
from ml_logic.model import run_model_workflow

def main():
    # Define the path to your raw data file
    data_filepath = 'path_to_your_raw_data.csv'  # Change this to your actual file path

    # Preprocess the data using the preprocess_data function from preprocessor.py
    print("Preprocessing the data...")
    df_encoded = preprocess_data(data_filepath)
    print("Data preprocessing complete.")

    # Run the model workflow using the run_model_workflow function from model.py
    print("Training and evaluating the model...")
    clf, accuracy = run_model_workflow(df_encoded)
    print(f"The model's accuracy is {accuracy * 100:.2f}%.")

if __name__ == "__main__":
    main()
