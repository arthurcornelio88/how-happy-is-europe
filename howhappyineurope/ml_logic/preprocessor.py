#libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
with open("features_table.json", 'r') as file:
    features_tables = json.load(file)
from data import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# from main_local

# def num_replacer(df): #, col):
#     for col in df.columns:
#         df_transformed = df.copy()
#         #df_transformed[f'{col}_refusal'] = df_transformed[col].apply(lambda x: 1 if x == 77 else -1)
#         df_transformed[f'{col}_dontknow'] = df_transformed[col].apply(lambda x: 1 if x == 88 else -1)
#         df_transformed[f'{col}_noanswer'] = df_transformed[col].apply(lambda x: 1 if x == 99 else -1)
#         # Replace values 77, 88, 99 with -1
#         df_transformed[col] = df_transformed[col].replace([77, 88, 99], -1)
#     return df_transformed

def smote_sampling(cleaned_data:pd.DataFrame)-> pd.DataFrame:
    categorical_features_indices = [cleaned_data.columns.get_loc("cntry")]
    smote_nc = SMOTENC(categorical_features=categorical_features_indices, random_state=42)
    y_reduced = cleaned_data['happy_reduced']
    cols = [col for col in cleaned_data.columns if "_desc" not in col and "happy" not in col]
    X_res, y_res = smote_nc.fit_resample(cleaned_data[cols], cleaned_data["happy_reduced"])
    cleaned_data = pd.concat([X_res, y_res], axis=1)
    return cleaned_data

def rescaling(cleaned_data, features_tables, json_to_df, enrich_df, remove_star_vals, create_map):
    features_map_dict = {}
    for feature in cleaned_data.columns:
        if feature in ['cntry', "happy_reduced"]:
            continue
        if feature == "wrklong":
            cleaned_data.loc[cleaned_data[feature] == 55, feature] = 66
        aux_df = json_to_df(features_tables, feature)
        cleaned_data = enrich_df(cleaned_data, aux_df)
        cleaned_data = remove_star_vals(cleaned_data, feature)

        mask = cleaned_data[feature+"_desc"] == "-1"
        vals = np.sort(cleaned_data[~mask][feature].unique())
        if 0 in vals:
            cleaned_data.loc[~mask, feature] += 1
            vals += 1
        map_dict = create_map(vals)
        features_map_dict[feature] = map_dict
        cleaned_data[feature] = cleaned_data[feature].replace(map_dict)
        cleaned_data.loc[mask, feature] = 0
        if len(vals)%2 == 1:
            indices = cleaned_data[cleaned_data[feature] == 1].index.to_list()
            indices = np.random.choice(indices, len(indices)//2)
            cleaned_data.loc[cleaned_data.index.isin(indices), feature] = -1
    cleaned_data = cleaned_data.reset_index(drop=True)
    return cleaned_data

def encoding_categorical_features(cleaned_data):
    encoder = OneHotEncoder(sparse_output=False)
    cntry_encoded = encoder.fit_transform(cleaned_data[['cntry']])
    encoded_columns = [f"cntry_{category}" for category in encoder.categories_[0]]
    cntry_encoded_df = pd.DataFrame(cntry_encoded, columns=encoded_columns)
    cleaned_data = pd.concat([cleaned_data.drop("cntry", axis=1), cntry_encoded_df], axis=1)
    return cleaned_data

def scaling(cleaned_data):
    minmax_X = MinMaxScaler()
    minmax_Y = MinMaxScaler()
    continuous_cols = [col for col in cleaned_data.columns if "_desc" not in col and "happy" not in col and "cntry" not in col]
    cntry_cols = [col for col in cleaned_data.columns if "cntry" in col]
    cleaned_data[continuous_cols] = minmax_X.fit_transform(cleaned_data[continuous_cols])
    cleaned_data["happy_reduced"] = minmax_Y.fit_transform(cleaned_data["happy_reduced"].values[:, np.newaxis])
    return cleaned_data, continuous_cols, cntry_cols

def split(cleaned_data, continuous_cols, cntry_cols):
    X = cleaned_data[continuous_cols + cntry_cols]
    y = cleaned_data["happy_reduced"].values[:, np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reutrn X_train, X_test, y_train, y_test

def pipe_preprocess(cleaned_data:pd.DataFrame)-> np.array:
    smote_sampling(cleaned_data)
    rescaling(cleaned_data, features_tables, json_to_df, enrich_df, remove_star_vals, create_map)
    encoding_categorical_features(cleaned_data)
    scaling(cleaned_data)
    split(cleaned_data, continuous_cols, cntry_cols)
    # Defining transformers (numerical and categorical)
    #num_replacer_transformer = FunctionTransformer(num_replacer)
    # cat_transformer = OneHotEncoder(handle_unknown='ignore',drop='first', sparse_output=False)
    # # Pipeline for processing data (preproc)
    # preproc = make_column_transformer(
    #     (cat_transformer, ['cntry','ctzcntr','brncntr','gndr', 'dscrgrp'])
    #     remainder='passthrough')
    # creating dataset processed (X_preproc)

    #X_preproc = preproc.fit_transform(clean_data)
    return X_train, X_test, y_train, y_test

    print("✅ preprocess() done")
    #return preproc, X_preproc
