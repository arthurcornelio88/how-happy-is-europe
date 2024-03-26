import pandas as pd
import numpy as np


def load_data(filepath):
    """Loads data from a file path."""
    df = pd.read_csv(filepath)
    # Add any initial cleaning steps here
    return df
