#main local package file

#libraries
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline


# import linear regression model
from sklearn.linear_model import LinearRegression

#hello
# Row for prediction
X_PRED = pd.DataFrame([np.array(["FR",1,1,1,1,1,6,1,1,1,1])], columns=['cntry', \
    'gndr', 'sclmeet', 'inprdsc', 'sclact', 'health', 'rlgdgr','dscrgrp',     \
    'ctzcntr', 'brncntr', 'happy'])

STATE_OF_HAPPINESS = [
    "Extremely unhappy",
    "Extremely unhappy",
    "Extremely unhappy",
    "Extremely unhappy",
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral",
    "Extremely happy",
    "Extremely happy",
    "Extremely happy"
]

# TODO:
# def getting_best_features():
#     return df_best_features

# from sklearn.linear_model import Ridge

def data_cleaning(df):

    # remove the first column
    df_cleaned = df.drop(df.columns[0], axis=1)
    return df_cleaned

def load_data():

    #load the basemodel dataset
    df = pd.read_csv("arthurcornelio88-notebooks/20240319_ESS10_baseline-model_arthurcornelio88.csv")
    return df

def num_replacer(df): #, col):
    for col in df.columns:
        df_transformed = df.copy()
        df_transformed[f'{col}_refusal'] = df_transformed[col].apply(lambda x: 1 if x == 77 else -1)
        df_transformed[f'{col}_dontknow'] = df_transformed[col].apply(lambda x: 1 if x == 88 else -1)
        df_transformed[f'{col}_noanswer'] = df_transformed[col].apply(lambda x: 1 if x == 99 else -1)
        # Replace values 77, 88, 99 with -1
        df_transformed[col] = df_transformed[col].replace([77, 88, 99], -1)
    return df_transformed

def pipe_preprocess(clean_data:pd.DataFrame)-> np.array:

    # Defining transformers (numerical and categorical)
    num_replacer_transformer = FunctionTransformer(num_replacer)
    cat_transformer = OneHotEncoder(handle_unknown='ignore',drop='first', sparse_output=False)
    # Pipeline for processing data (preproc)
    preproc = make_column_transformer(
        (num_replacer_transformer, ['sclmeet','inprdsc','health', 'rlgdgr']),
        (cat_transformer, ['cntry','ctzcntr','brncntr','gndr', 'dscrgrp']),
        remainder='passthrough')
    # creating dataset processed (X_preproc)
    X_preproc = preproc.fit_transform(clean_data)

    print("✅ preprocess() done")
    return preproc, X_preproc

def train(df_processed):
    # no need for evaluation because it was already done in notebook experimentation
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X, y = df_processed[:, :-1], df_processed[:, -1]

    # TODO : maybe when model enhanced
    #1) build evaluate_model function in model.py
    # metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    #2) set params
    # params = dict(
    #     context="train",
    #     training_set_size=DATA_SIZE,
    #     row_count=len(X_train_processed),
    #3) already done in registry.py
    #save_results(params=params, metrics=metrics_dict)


    # Training the linear regression model
    # TODO : create model.py
    #model = load_model()
    model = LinearRegression()
    model = model.fit(X, y)

    print("✅ train() done")
    return model

def pred(X_pred):

    df = load_data()
    df_cleaned = data_cleaning(df)
    #df_best = getting_best_features(df_cleaned)

    # Creating pipeline for processing data (preproc) and
        # creating dataset processed (X_preproc)
    preproc, X_preproc = pipe_preprocess(df_cleaned)

    #saving process pipeline as pickle file
    with open("pipeline.pkl", "wb") as file:
        pickle.dump(preproc, file)

    # Training model
    model = train(X_preproc)

    # TODO : choose method of saving
    #saving trained model as pickle file
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)
    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    # TODO : when we have the enhanced model
    # Save results on the hard drive using taxifare.ml_logic.registry
    #save_results(params=params, metrics=dict(mae=val_mae))



    # Processing the test data
    x_pred_preproc = preproc.transform(X_pred)

    # Making the prediction
    y_pred = model.predict(x_pred_preproc[:, :-1])[0]

    # Rounding the predictions to the nearest integer and constraining them to the range [0, 10]
    y_pred_constrained = int(np.clip(np.round(y_pred), 0, 10))

    print(y_pred_constrained)
    print(f"You are {STATE_OF_HAPPINESS[y_pred_constrained]}." )
    #return y_pred_constrained


if __name__ == '__main__':
    pred(X_PRED)
