from sklearn.linear_model import LogisticRegression


def get_model(max_iter):
    return LogisticRegression(max_iter=max_iter)
