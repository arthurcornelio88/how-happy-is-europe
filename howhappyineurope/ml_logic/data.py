import numpy as np
import pandas as pd
import json
from typing import Dict

def load_data():
    # Maybe, in the future, getting data from a filtering in bigquery, or
    #the POST to get 1000 specific rows, etc.
    # TODO DONE
    #load the pre-filtered dataset
    data = pd.read_csv(
        "../data/20240319_ESS10_manually-filtered_arthurcornelio88.csv"
    ).reset_index(drop=True)
    #load the dictionary with the best features
    with open("../data/features_table.json", 'r') as file:
        features_tables = json.load(file)

    return data, features_tables

def data_cleaning(data):
    # TODO: DONE
    # clean undesired rows and the target "happy"
    data_cleaned = data.drop("Unnamed: 0", axis=1)
    data_cleaned = data[~data["happy"].isin([77, 88, 99])]
    return data_cleaned

def json_to_df(features_json: Dict, feature: str) -> pd.DataFrame:
    """
    Converts a specified feature from a JSON object into a pandas DataFrame. The function expects the feature to be
    represented as a dictionary within the JSON object, where each key-value pair corresponds to an entity's identifier
    and its description, respectively. It creates a DataFrame with two columns: one for the identifiers (appending '_key'
    to the feature name) and one for the descriptions (appending '_desc' to the feature name), ensuring that identifiers
    are of integer type and descriptions are of string type.

    Parameters:
    -----------
    df : pd.DataFrame
        The original DataFrame to be enriched.
    enrichment_df : pd.DataFrame
        The DataFrame containing additional information for the enrichment, with columns named 'feature_key' and 'feature_desc'.

    Returns:
    -------
    pd.DataFrame
        The enriched DataFrame with additional descriptive information merged on the specified feature.
    """
    feature_dict = features_json[feature]
    df = pd.DataFrame.from_dict(
        feature_dict,
        columns=["val"],
        orient="index").reset_index()
    df.columns = [feature+"_key", feature+"_desc"]
    df[feature+"_key"] = df[feature+"_key"].astype(int)
    df[feature+"_desc"] = df[feature+"_desc"].astype(str)
    return df

def enrich_df(df: pd.DataFrame, enrichment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches a given DataFrame by merging it with another DataFrame that contains additional information for one of its
    features. The function automatically identifies the feature to be enriched based on the naming convention used in the
    'enrichment_df' (expects 'feature_key'). It performs a left merge on the specified feature, effectively adding the
    descriptive information from 'enrichment_df' into 'df' and dropping the redundant key column from the merge.

    Parameters:
    -----------
    df : pd.DataFrame
        The original DataFrame to be enriched.
    enrichment_df : pd.DataFrame
        The DataFrame containing additional information for the enrichment, with columns named 'feature_key' and 'feature_desc'.

    Returns:
    -------
    pd.DataFrame
        The enriched DataFrame with additional descriptive information merged on the specified feature.
    """
    feature = enrichment_df.columns[0].replace("_key", "")
    return df.merge(
        enrichment_df,
        how="left",
        left_on=feature,
        right_on=feature+"_key"
    ).drop(feature+"_key", axis=1)

def remove_star_vals(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Cleans the input DataFrame by removing rows where the description of a specified feature is marked with an asterisk ("*").
    It calculates and prints the percentage of rows being removed because of this placeholder value. This step is crucial for
    cleaning the dataset by eliminating entries with non-informative placeholder values, ensuring data quality for further
    analysis or modeling.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to be cleaned.
    feature : str
        The feature whose descriptions are to be examined for placeholder values.

    Returns:
    -------
    pd.DataFrame
        A DataFrame excluding rows with placeholder descriptions for the specified feature.
    """
    mask = df[feature+"_desc"] == "*"
    print(f"{feature} %rows: " + str(len(df[mask]) / len(df) * 100))
    return df[~mask]

def create_map(arr: np.ndarray) -> Dict:
    aux_vals = np.arange(1, arr[-1] // 2 + 1)
    if len(arr)%2 == 0:
        aux_vals = np.concatenate([-np.flip(aux_vals), aux_vals])
    else:
        aux_vals = np.concatenate([-np.flip(aux_vals) - 1, np.array([1]), aux_vals + 1])
    dict_map = {}
    for ind, el in enumerate(arr):
        dict_map[el] = aux_vals[ind]
    return dict_map
