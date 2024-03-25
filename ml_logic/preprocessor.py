import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# Data loading and initial preprocessing
def preprocess_target(df,features):
    # features = ['cntry', 'gndr', 'sclmeet', 'inprdsc', 'sclact', 'health', 'rlgdgr', 'dscrgrp', 'ctzcntr', 'brncntr', 'happy']
    base_df = df[features]
    mask = base_df["happy"].isin([77, 88, 99])
    base_df = base_df[~mask].reset_index(drop=True)
    return base_df



def feature_scale_map(df, feature):
    map_dict = {}

    all_vals = np.sort(df[feature].unique())

    if np.any(all_vals[1:] - all_vals[0:-1] > 1):
        min_ind = np.where(all_vals[1:] - all_vals[0:-1] > 1)[0][0]
        vals = all_vals[:min_ind].copy()
        min_val = vals.min()
        if min_val == 1:
            vals = vals - 1
        for i, val in enumerate(all_vals[:-1]):
            map_dict[val] = i
        map_dict[all_vals[-1]] = -1
        df[feature] = df[feature].replace(map_dict)
    else:
        for i, val in enumerate(all_vals):
            map_dict[val] = i
        df[feature] = df[feature].replace(map_dict)


def scale_features(df, columns):
    """Scales features using MinMaxScaler."""
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns, index=df.index)
    return scaled_df


def encode_categorical(df, categorical_features):
    """Encodes categorical features using OneHotEncoder."""
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df[categorical_features]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))
    # Drop the original categorical columns and concatenate the new one-hot encoded columns
    df = df.drop(categorical_features, axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df
