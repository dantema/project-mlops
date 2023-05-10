import pandas as pd
from sklearn.model_selection import train_test_split

def read_prepare_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = pd.read_csv(path)

    df.loc[df["model"] == "Focus", "model"] = " Focus"
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    return df_train, df_test

