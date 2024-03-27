import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from howhappyineurope.params import STATE_OF_HAPPINESS, ROOT_DIR, CONT_COLS, CATEG_COLS
import joblib as jb

app = FastAPI()
app.state.model = jb.load(f"{ROOT_DIR}/ml_logic/ml_obj/model.joblib")
app.state.minmax_x = jb.load(f"{ROOT_DIR}/ml_logic/ml_obj/minmax_scalar_x.joblib")
app.state.minmax_y = jb.load(f"{ROOT_DIR}/ml_logic/ml_obj/minmax_scalar_y.joblib")
app.state.onehotencoder = jb.load(f"{ROOT_DIR}/ml_logic/ml_obj/one_hot_encoder.joblib")

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict
@app.get("/predict")
def predict(
    cntry: str,
    gndr: int, stfmjob: int, dcsfwrka: int, wrkhome: int, wrklong: int,
    wrkresp: int, sclmeet: int, sclact: int, trdawrk: int, jbprtfp: int,
    pfmfdjba: int, health: int, hlthhmp: int, hhmmb: int, hincfel: int,
    stfeco: int, ipsuces: int, iphlppl: int, ipstrgv: int, trstplc: int
    ):
    """
    list input from user to predict their happiness. list of features needed for prediction.
    """

    # !!! Order in website is different from the code dataset
    # "cntry","stfmjob","trdawrk","jbprtfp", "pfmfdjba", "dcsfwrka", "wrkhome",
    # "wrklong", "wrkresp", "health","stfeco","hhmmb","hincfel", "trstplc",
    # "sclmeet", "hlthhmp", "sclact","iphlppl", "ipsuces", "ipstrgv", "gndr"

    df = pd.DataFrame(dict(
        cntry=cntry,
        stfmjob=stfmjob,trdawrk=trdawrk,jbprtfp=jbprtfp,pfmfdjba=pfmfdjba,
        dcsfwrka=dcsfwrka,wrkhome=wrkhome,wrklong=wrklong,wrkresp=wrkresp,
        health=health,stfeco=stfeco,hhmmb=hhmmb,hincfel=hincfel,trstplc=trstplc,
        sclmeet=sclmeet,hlthhmp=hlthhmp,sclact=sclact,iphlppl=iphlppl,
        ipsuces=ipsuces,ipstrgv=ipstrgv,gndr=gndr
    ), index=[0])

    model = app.state.model
    minmax_x = app.state.minmax_x
    minmax_y = app.state.minmax_y
    onehotencoder = app.state.onehotencoder

    x_transformed = minmax_x.transform(df[CONT_COLS])
    cntry_transformed = onehotencoder.transform(df[CATEG_COLS])
    x_transformed = np.concatenate([x_transformed, cntry_transformed], axis=1)
    y_pred = model.predict(x_transformed)[:, np.newaxis]
    y_pred = np.round(minmax_y.inverse_transform(y_pred))
    number_y_pred = int(y_pred[0][0])

    print("\n✅ prediction done: ", number_y_pred, "\n")
    prediction_done = f'You are {STATE_OF_HAPPINESS[number_y_pred]}'
    return prediction_done

@app.get("/")
def root():
    return dict(greeting="Hello")
