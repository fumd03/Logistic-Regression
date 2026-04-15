# data.py

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(test_size, random_state):
    # Load CSV
    df = pd.read_csv("../data/hearing_test.csv")

    # Features (X) and target (y)
    X = df[["age", "physical_score"]]
    y = df["test_result"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
