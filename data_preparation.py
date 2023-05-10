import pandas as pd
from sklearn.model_selection import train_test_split
from params import split_params

def read_prepare_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df_train, df_test = train_test_split(df, **split_params)
    return df_train, df_test
