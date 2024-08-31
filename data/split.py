import pandas as pd
from sklearn.model_selection import train_test_split


# Splits the dataset into train and test
def split():
    data = pd.read_csv("imdb.csv")

    X = data.review
    y = data.sentiment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    df_train.to_csv("./imdb_original_train.csv", index=False)
    df_test.to_csv("./imdb_test.csv", index=False)


split()
