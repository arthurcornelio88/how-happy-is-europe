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
X_PRED = pd.DataFrame([np.array(["FR",4,3,2,5,7,6,2,4,6])], columns=['cntry', \
    'gndr', 'sclmeet', 'inprdsc', 'sclact', 'health', 'rlgdgr','dscrgrp',     \
    'ctzcntr', 'brncntr'])

# TODO:
STATE_OF_HAPPINESS = [
    "",
    "",
]
def data_cleaning(x):

    # remove the first column
    df = df.drop(df.columns[0], axis=1)
    return x

def load_data():

    #load the basemodel dataset
    df = pd.read_csv("arthurcornelio88-notebooks/20240319_ESS10_baseline-model_arthurcornelio88.csv")
    df_cleaned = data_cleaning(df)
    ###Seperate features and target
    return df_cleaned

def preprocess(clean_data:pd.DataFrame)-> np.array:

    #One hot encoding of cntry, ctzcntr, brncntr, gndr, dscrgrp
    X = pd.get_dummies(clean_data, columns=['cntry'], drop_first=True)
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

    print("✅ preprocess_and_train() done")
    return X


def train(df):
    # no need for evaluation because it was already done in notebook experimentation
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # Training the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    return model

def pred(X_pred):

    df = load_data()
    X_preproc = preprocess(df)
    # Making predictions
    model = train(X_preproc)
    y_pred = model.predict(X_pred)

    # Rounding the predictions to the nearest integer and constraining them to the range [0, 10]
    y_pred_constrained = np.clip(np.round(y_pred), 0, 10)

    print(f"You are {STATE_OF_HAPPINESS[y_pred_constrained]}." )
    return y_pred_constrained


if __name__ == '__main__':
    pred(X_PRED)
