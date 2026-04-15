import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path, test_size, random_state):
    df = pd.read_csv(path)

    X = df[["age", "physical_score"]]
    y = df["test_result"]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
