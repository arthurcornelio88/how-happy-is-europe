import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(
        cntry: str, # FR
        gndr: int, # 4
        sclmeet: int, # 3
        inprdsc: int, # 2
        sclact: int, # 5
        health: int, # 7
        rlgdgr: int, # 6
        dscrgrp: int, # 2
        ctzcntr: int, # 4
        brncntr: int
    ):      # 6
    """
    list input from user to predict their happiness. list of features needed for prediction.
    """

    # TODO: which one to assign X_pred ?
    X_pred = pd.DataFrame(locals(), index=[0])
    #OR
    X_pred = pd.DataFrame(dict(
        pickup_datetime=pickup_datetime,
        pickup_longitude=pickup_longitude,
        pickup_latitude=pickup_latitude,
        dropoff_longitude=dropoff_longitude,
        dropoff_latitude=dropoff_latitude,
        passenger_count=passenger_count,
        ), index=[0])

    # load the model - store it in 'model' folder. api will access folder and load it from there. put this file inside docker image
    # TODO : function in registry.py? that loads the pickle in model.py? because model is already trained !
    model = app.state.model
    assert model is not None

    # preprocess input data
    # TODO: create a preprocess_features.py that will handle that
        # x_pred_preproc = preproc.transform(X_pred)
    x_pred_preproc = preprocess_features(X_pred)


    # Making the prediction
    y_pred = model.predict(x_pred_preproc[:, :-1])[0]

    # Rounding the predictions to the nearest integer and constraining them to the range [0, 10]
    y_pred_constrained = int(np.clip(np.round(y_pred), 0, 10))

    # TODO: create the STATE_OF_HAPPINESS variable in .env or elsewhere
    print(f"You are {STATE_OF_HAPPINESS[y_pred_constrained]}." )
    #return y_pred_constrained

    # TODO: ameliorate the display of info
    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    prediction_done = f'0 to 10: {y_pred_constrained} - {STATE_OF_HAPPINESS[y_pred_constrained]}'

    return dict(fare_amount=str(prediction_done))
