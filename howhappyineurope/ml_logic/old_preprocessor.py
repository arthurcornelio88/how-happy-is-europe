#libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline

# from main_local

def num_replacer(df): #, col):
    for col in df.columns:
        df_transformed = df.copy()
        df_transformed[f'{col}_refusal'] = df_transformed[col].apply(lambda x: 1 if x == 77 else -1)
        df_transformed[f'{col}_dontknow'] = df_transformed[col].apply(lambda x: 1 if x == 88 else -1)
        df_transformed[f'{col}_noanswer'] = df_transformed[col].apply(lambda x: 1 if x == 99 else -1)
        # Replace values 77, 88, 99 with -1
        df_transformed[col] = df_transformed[col].replace([77, 88, 99], -1)
    return df_transformed

def pipe_preprocess(clean_data:pd.DataFrame)-> np.array:

    # Defining transformers (numerical and categorical)
    num_replacer_transformer = FunctionTransformer(num_replacer)
    cat_transformer = OneHotEncoder(handle_unknown='ignore',drop='first', sparse_output=False)
    # Pipeline for processing data (preproc)
    preproc = make_column_transformer(
        (num_replacer_transformer, ['sclmeet','inprdsc','health', 'rlgdgr']),
        (cat_transformer, ['cntry','ctzcntr','brncntr','gndr', 'dscrgrp']),
        remainder='passthrough')
    # creating dataset processed (X_preproc)
    X_preproc = preproc.fit_transform(clean_data)

    print("✅ preprocess() done")
    return preproc, X_preproc
