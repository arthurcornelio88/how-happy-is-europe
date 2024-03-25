#main package file

from howhappyineurope.ml_logic.registry import *
from howhappyineurope.ml_logic.model import *
from howhappyineurope.ml_logic.data import *
from howhappyineurope.ml_logic.preprocessor import *
from howhappyineurope.params import *

#libraries
import numpy as np
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline

# Row for prediction, already on params.py
X_PRED = pd.DataFrame([np.array(["FR",1,1,1,1,1,6,1,1,1,1])], columns=['cntry', \
    'gndr', 'sclmeet', 'inprdsc', 'sclact', 'health', 'rlgdgr','dscrgrp',     \
    'ctzcntr', 'brncntr', 'happy'])

def train(X_train, y_train):

    # TODO: now
    # Train model using `model.py`
    model = load_model()

    if model is None:
        model = initialize_model()

    model = train_model(model, X_train, y_train)

    print("✅ train() done")
    return model

# TODO: when working on enhanced model
def evaluate(model, X_test, y_test):

    metrics_dict = evaluate_model(model, X_test, y_test)
    mae = metrics_dict["mae"]

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=DATA_SIZE,
        row_count=len(X_new)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae

def pred(X_pred=X_PRED):
    #import ipdb; ipdb.set_trace()
    df, features_table = load_data()
    df_cleaned = data_cleaning(df)
    df_reduce = reduce_happiness_categories(FEATURES_DICT, df_cleaned)

    df_preproc = pipe_preprocess(df_reduce)
    X_train, X_test, y_train, y_test = split(df_preproc)

    # Training model
    model = train(X_train)

    # TODO : choose method of saving

    with open("/home/arthurcornelio/code/arthurcornelio88/how-happy-in-europe/models/model.pkl", "wb") as file:
         pickle.dump(model, file)
    # CHOOSED: Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    # Processing the row data
    x_pred_preproc = scaling(X_pred)

    # Making the prediction
    y_pred = model.predict(x_pred_preproc[:, :-1])[0]

    print(y_pred)
    # TODO
    # evaluate
    # evaluate(model, x_test, y_test)

    # TODO: when we have the enhanced model
    # Save results on the hard drive using taxifare.ml_logic.registry
    #save_results(params=params, metrics=dict(mae=val_mae))

    # Rounding the predictions to the nearest integer and constraining them to the range [0, 10]
    y_pred_constrained = int(np.clip(np.round(y_pred), 0, 10))

    print(y_pred_constrained)
    print(f"You are {STATE_OF_HAPPINESS[y_pred_constrained]}." )
    #return y_pred_constrained

if __name__ == '__main__':
    pred(X_PRED)
