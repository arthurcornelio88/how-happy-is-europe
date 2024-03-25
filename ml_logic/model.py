from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

def split_data(df_encoded, target_column='happy', test_size=0.2, random_state=0):
    """Splits the data into training and test sets."""
    X = df_encoded.drop(target_column, axis=1).copy()
    y = df_encoded[target_column].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, num_class):
    """Trains the XGBoost classifier."""
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_class,
        eval_metric='mlogloss',
    )
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """Evaluates the trained classifier on the test set."""
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# This function might be called from your main script, after preprocessing the data.
def run_model_workflow(df_encoded):
    """Runs the complete model training and evaluation workflow."""
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_encoded)

    # Train model
    num_class = len(set(y_train)) # This can also be defined globally or passed as an argument
    clf = train_model(X_train, y_train, num_class)

    # Evaluate model
    accuracy = evaluate_model(clf, X_test, y_test)

    return clf, accuracy  # You might want to return the model or accuracy if needed elsewhere
