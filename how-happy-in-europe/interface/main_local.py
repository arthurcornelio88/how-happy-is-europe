#main local package file

#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Row for prediction
X_PRED = pd.DataFrame([np.array(["FR",1,1,1,1,1,6,1,1,1])], columns=['cntry', \
    'gndr', 'sclmeet', 'inprdsc', 'sclact', 'health', 'rlgdgr','dscrgrp',     \
    'ctzcntr', 'brncntr'])

# TODO: 0-3 extremely unhappy, 4-7:neutral, 8-10:extremely happy
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
def data_cleaning(df):

    # remove the first column
    df_cleaned = df.drop(df.columns[0], axis=1)
    return df_cleaned

def load_data():

    #load the basemodel dataset
    df = pd.read_csv("arthurcornelio88-notebooks/20240319_ESS10_baseline-model_arthurcornelio88.csv")

    return df

def preprocess(clean_data:pd.DataFrame, pred:int, cleaned_full_data:pd.DataFrame=None)-> np.array:

    # TODO : can I do this ? I`m just replacing the row by all dataset but I want to treat row with the columns from the big one
    #get dummies...
    if pred == 1 :
        clean_data = clean_data
    elif pred == 0:
        clean_data = cleaned_full_data

    X = clean_data
    #One hot encoding of cntry, ctzcntr, brncntr, gndr, dscrgrp
    X = pd.get_dummies(X, columns=['cntry'], drop_first=True)
    X = pd.get_dummies(X, columns=['ctzcntr'], drop_first=True)
    X = pd.get_dummies(X, columns=['brncntr'], drop_first=True)
    X = pd.get_dummies(X, columns=['gndr'], drop_first=True)
    X = pd.get_dummies(X, columns=['dscrgrp'], drop_first=True)

    # For sclmeet, create a new column called sclmeet_refusal, sclmeet_dontknow, sclmeet_noanswer , put true if the value is 77, 88, 99 respectively
    X['sclmeet_refusal'] = X['sclmeet'].apply(lambda x: 1 if x == 77 else 0)
    X['sclmeet_dontknow'] = X['sclmeet'].apply(lambda x: 1 if x == 88 else 0)
    X['sclmeet_noanswer'] = X['sclmeet'].apply(lambda x: 1 if x == 99 else 0)
    # Replace values 77, 88, 99 with O
    X['sclmeet'] = X['sclmeet'].replace([77, 88, 99], 0)

    # Same for inprdsc
    X['inprdsc_refusal'] = X['inprdsc'].apply(lambda x: 1 if x == 77 else 0)
    X['inprdsc_dontknow'] = X['inprdsc'].apply(lambda x: 1 if x == 88 else 0)
    X['inprdsc_noanswer'] = X['inprdsc'].apply(lambda x: 1 if x == 99 else 0)
    # Replace values 77, 88, 99 with O
    X['inprdsc'] = X['inprdsc'].replace([77, 88, 99], 0)

    # Same for health
    X['health_refusal'] = X['health'].apply(lambda x: 1 if x == 77 else 0)
    X['health_dontknow'] = X['health'].apply(lambda x: 1 if x == 88 else 0)
    X['health_noanswer'] = X['health'].apply(lambda x: 1 if x == 99 else 0)
    # Replace values 77, 88, 99 with O
    X['health'] = X['health'].replace([77, 88, 99], 0)

    # Same for rlgdgr
    X['rlgdgr_refusal'] = X['rlgdgr'].apply(lambda x: 1 if x == 77 else 0)
    X['rlgdgr_dontknow'] = X['rlgdgr'].apply(lambda x: 1 if x == 88 else 0)
    X['rlgdgr_noanswer'] = X['rlgdgr'].apply(lambda x: 1 if x == 99 else 0)
    # Replace values 77, 88, 99 with O
    X['rlgdgr'] = X['rlgdgr'].replace([77, 88, 99], 0)

    #Replace all True with 1 and False with 0
    X = X.replace([True, False], [1, 0])
    df_processed = X

    print("✅ preprocess_and_train() done")

    return df_processed

def train(df_processed):
    # no need for evaluation because it was already done in notebook experimentation
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X, y = df_processed.iloc[:, :-1], df_processed.iloc[:, -1]
    # Training the linear regression model
    model = LinearRegression()
    model = model.fit(X, y)

    return model

def pred(X_pred):

    df = load_data()
    df_cleaned = data_cleaning(df)
    breakpoint()
    X_preproc = preprocess(df_cleaned,1)

    # Making predictions
    model = train(X_preproc)

    #preprocessing the survey-answers to be predicted (one row, one person)
    #with the full dataset and then just taking
    x_pred_preproc = preprocess(X_pred,0,df_cleaned).iloc[:,:-1]

    #y_pred = model.predict(x_pred_preproc)
    y_pred = model.predict(X_PRED)

    # Rounding the predictions to the nearest integer and constraining them to the range [0, 10]
    y_pred_constrained = np.clip(np.round(y_pred), 0, 10)

    print(y_pred_constrained)
    print(f"You are {STATE_OF_HAPPINESS[y_pred_constrained]}." )
    #return y_pred_constrained


if __name__ == '__main__':
    pred(X_PRED)
