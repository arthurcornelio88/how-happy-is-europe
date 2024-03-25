#main package file

from howhappyineurope.ml_logic.registry import *
from howhappyineurope.ml_logic.model import *
from howhappyineurope.ml_logic.data import *
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

def train(df_processed):
    #split processed data into X and y
    X, y = df_processed[:, :-1], df_processed[:, -1]

    # TODO: after d_image baseline, do it in model.py
    # 0) for now, no need for evaluation because it was already done in notebook experimentation
    ### CODE HERE:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #### CODE HERE:
    # metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    # params = dict(
    #     context="train",
    #     training_set_size=DATA_SIZE,
    #     row_count=len(X_train_processed),
    ### already done in registry.py
    #save_results(params=params, metrics=metrics_dict)

    # TODO: now
    # Train model using `model.py`
    model = load_model()

    if model is None:
        model = initialize_model(input_shape=df_processed.shape[1:])

    ### When Neural Networks, for now, no need for compiling
    #model = compile_model(model, learning_rate=learning_rate)
    model = train_model(model, X, y)

    #==============ANCIEN
    # model = LinearRegression()
    # model = model.fit(X, y)
    #==============ANCIEN
    print("✅ train() done")
    return model

# TODO: when working on enhanced model
#def evaluate():
    model = load_model(stage=stage)
    assert model is not None

    ###GET DATA

    metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
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

    df, features_table = load_data()
    df_cleaned = data_cleaning(df)
    #df_best = getting_best_features(df_cleaned)

    # Creating pipeline for processing data (preproc) and
        # creating dataset processed (X_preproc)
    preproc, X_preproc = pipe_preprocess(df_cleaned)

    #saving process pipeline as pickle file
    with open("/home/arthurcornelio/code/arthurcornelio88/how-happy-in-europe/pipelines/pipeline.pkl", "wb") as file:
        pickle.dump(preproc, file)

    # Training model
    model = train(X_preproc)

    # TODO : choose method of saving
    #saving trained model as pickle file
    # with open("/home/arthurcornelio/code/arthurcornelio88/how-happy-in-europe/models/model.pkl", "wb") as file:
    #     pickle.dump(model, file)
    # CHOOSED: Save model weight on the hard drive (and optionally on GCS too!)
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
