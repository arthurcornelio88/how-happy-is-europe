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

#def train(): # TODO

#     # TODO:
#     df = load_data()
#     df = data_cleaning(df)
#     df = reduce_happiness_categories(FEATURES_DICT, df)

#     df_preproc = pipe_preprocess(df)
#     X_train, X_test, y_train, y_test = split(df_preproc)

#     model = train_model(X_train, y_train)

#     # evaluate
#     evaluate_model(model, X_test, y_test)
#     # TODO: now
#     # Train model using `model.py`
#     model = load_model()

#     if model is None:
#         model = initialize_model()

#     model = train_model(model, X_train, y_train)

#     save_model(model)

#     print("✅ train() done")
#     return model

def fun1(): #put or not in the pipeline?
    data = data.drop("Unnamed: 0", axis=1)
    data = data[~data["happy"].isin([77, 88, 99])]

# def pipe_preprocess(clean_data:pd.DataFrame)-> np.array:

#     #? fun1,

#     (num_replacer_transformer, ['sclmeet','inprdsc','health', 'rlgdgr']),
#     (cat_transformer, ['cntry','ctzcntr','brncntr','gndr', 'dscrgrp']),

#     # Defining transformers (numerical and categorical)
#     num_replacer_transformer = FunctionTransformer(num_replacer)
#     cat_transformer = OneHotEncoder(handle_unknown='ignore',drop='first', sparse_output=False)
#     # Pipeline for processing data (preproc)
#     preproc = make_column_transformer(
#         (num_replacer_transformer, ['sclmeet','inprdsc','health', 'rlgdgr']),
#         (cat_transformer, ['cntry','ctzcntr','brncntr','gndr', 'dscrgrp']),
#         remainder='passthrough')
#     # creating dataset processed (X_preproc)
#     X_preproc = preproc.fit_transform(clean_data)

#     print("✅ preprocess() done")
#     return preproc, X_preproc

def pred(X_pred=X_PRED):

    # Processing the row to predict data
    X_pred_main = X_pred

    print(X_pred)

    for column in X_pred.columns:
        if column != 'cntry':
            X_pred[column] = X_pred[column].astype(int)

    # aux_data = X_pred[FEATURES_DICT.keys()].copy()

    # reduce_class_map = {
    #     0: 0, 1: 0, 2: 0, 3: 0,
    #     4: 1, 5: 1, 6: 1, 7: 1,
    #     8: 2, 9: 2, 10: 2
    # }

    # aux_data["happy_reduced"] = aux_data["happy"].replace(reduce_class_map)
    # aux_data = aux_data.reset_index(drop=True)
    X_original = load_data()
    X_pred_main = rescaling(X_pred)
    print(X_pred_main)
    X_pred_main, encoder = encoding_categorical_features(X_pred_main,predicting=False)
    print(X_pred_main)
    print(X_pred_main.columns)
    X_pred_main = scaling(X_pred_main)
    print(X_pred_main.columns)
    print(X_pred_main)
    model = load_model()
    # Making the prediction
    y_pred = model.predict(X_pred_main)

    # Post process
    print(y_pred)

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
