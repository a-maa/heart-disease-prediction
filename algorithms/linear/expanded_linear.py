import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score


def load_dataset(path):
    return pd.read_csv(path)


def preprocess(data):
    return pd.get_dummies(data) #one-hot encoding for categorical features


def expand_features(data):
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    #expand features using other features only - avoid extra cost
    return poly.fit_transform(data)


def train_model(X, y, k=5):
    clsf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    accuracies = []

    for train_index, test_index in skf.split(X, y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clsf.fit(x_train, y_train)
        y_pred = clsf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy) #accuracy list

    return accuracies


def main():
    data = load_dataset("heart.csv")

    processed_data = preprocess(data) #preprocessing

    x = processed_data.drop(columns=["HeartDisease"]).values
    y = processed_data["HeartDisease"].values #split features & labels

    x_panded = expand_features(x)  #expand features

    accuracies = train_model(x_panded, y, k=5) #train model

    print("Accuracies per fold:", accuracies)
    print("Mean Accuracy:", np.mean(accuracies))


if __name__ == "__main__":
    main()
