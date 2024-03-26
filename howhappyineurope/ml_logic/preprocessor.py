#libraries
import json
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import  OneHotEncoder
from sklearn.model_selection import train_test_split
from howhappyineurope.ml_logic.data import *
with open("features_table.json", 'r') as file:
    FEATURES_TABLE = json.load(file)
def smote_sampling(df:pd.DataFrame)-> pd.DataFrame:
    """
    get the input dataframe, balance each category of happiness
    using SMOTENC
    Arguments:
    ----------
    df: pd.DataFrame
        input dataframe with categorical feature "cntry"
    Returns:
    --------
    pd.DataFrame
        balanced dataframe for each category of "happy_reduced"
    """
    # get the index of the categorical column to be passed to SMOTENC
    categorical_features_indices = [df.columns.get_loc("cntry")]
    smote_nc = SMOTENC(categorical_features=categorical_features_indices, random_state=42)
    # get all the columns that are not descriptive and are not the target feature i.e. "happy_reduced"
    cols = [col for col in df.columns if "_desc" not in col and "happy" not in col]
    X_res, y_res = smote_nc.fit_resample(df[cols], df["happy_reduced"])
    # reconstruct the full dataframe by concatenating the independent and dependent features
    return pd.concat([X_res, y_res], axis=1)
def rescaling(df):
    for feature in df.columns:
        if feature in ['cntry', "happy_reduced"]:
            continue
        if feature == "wrklong":
            df.loc[df[feature] == 55, feature] = 66
        aux_df = json_to_df(FEATURES_TABLE, feature)
        df = enrich_df(df, aux_df)
        df = remove_star_vals(df, feature)
        mask = df[feature+"_desc"] == "-1"
        vals = np.sort(df[~mask][feature].unique())
        if 0 in vals:
            df.loc[~mask, feature] += 1
            vals += 1
        map_dict = create_map(vals)
        df[feature] = df[feature].replace(map_dict)
        df.loc[mask, feature] = 0
        if len(vals)%2 == 1:
            indices = df[df[feature] == 1].index.to_list()
            indices = np.random.choice(indices, len(indices)//2)
            df.loc[df.index.isin(indices), feature] = -1
    return df.reset_index(drop=True)
def encoding_categorical_features(df):
    encoder = OneHotEncoder(sparse_output=False)
    cntry_encoded = encoder.fit_transform(df[['cntry']])
    encoded_columns = [f"cntry_{category}" for category in encoder.categories_[0]]
    cntry_encoded_df = pd.DataFrame(cntry_encoded, columns=encoded_columns)
    return pd.concat([df.drop("cntry", axis=1), cntry_encoded_df], axis=1)
def scaling(df: pd.DataFrame):
    minmax_X = MinMaxScaler()
    minmax_Y = MinMaxScaler()
    continuous_cols = [col for col in df.columns if "_desc" not in col and "happy" not in col and "cntry" not in col]
    df[continuous_cols] = minmax_X.fit_transform(df[continuous_cols])
    df["happy_reduced"] = minmax_Y.fit_transform(df["happy_reduced"].values[:, np.newaxis])
    return df
def split(df):
    continuous_cols = [col for col in df.columns if "cntry" not in col and "happy" not in col]
    cntry_cols = [col for col in df.columns if "cntry" in col]
    X = df[continuous_cols + cntry_cols]
    y = df["happy_reduced"].values[:, np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
def pipe_preprocess(df:pd.DataFrame)-> np.array:
    df = rescaling(df)
    df = smote_sampling(df)
    df = encoding_categorical_features(df)
    df = scaling(df)
    print(":white_check_mark: preprocess() done")
    return df
