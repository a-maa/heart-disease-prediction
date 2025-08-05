import pandas as pd
import plotly.express as px
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


"""
This function loads the dataset from a csv file and returns it as a pandas DataFrame.

Parameters:
    type (str): The type of data to be returned. If "str", the data is returned as a string. Otherwise, the data is returned as a pandas DataFrame.
    path (str): The path to the csv file

Returns:
    data (pd.DataFrame): The dataset as a str (useful for plotting) or pandas DataFrame
"""


def load_dataset(type="default", path="./datasets/heart.csv") -> pd.DataFrame:
    data = pd.read_csv(path)

    if type == "str":
        return data.astype(str)
    elif type == "default":
        return data
    else:
        raise ValueError("Invalid type")


"""
This function transforms the columns of the dataset to be used in the model.

Parameters:
    data (pd.DataFrame): The dataset to be transformed

Returns:
    data (pd.DataFrame): The transformed dataset
"""


def transform_columns(data: pd.DataFrame) -> pd.DataFrame:
    data["Sex"] = data["Sex"].map({'M': 1, 'F': 0})
    data["ExerciseAngina"] = data["ExerciseAngina"].map({'Y': 1, 'N': 0})
    data["ChestPainType"] = data["ChestPainType"].map(
        {'ATA': 1, 'NAP': 2, 'ASY': 3, 'TA': 4})
    data["RestingECG"] = data["RestingECG"].map(
        {'Normal': 0, 'ST': 1, 'LVH': 2})
    data["ST_Slope"] = data["ST_Slope"].map({'Up': 1, 'Flat': 2, 'Down': 3})

    return data


"""
This function plots the correlation matrix of the dataset.

Parameters:
    data (pd.DataFrame): The dataset to be used

Returns:
    None
"""


def plot_correlation_matrix(data: pd.DataFrame) -> None:
    correlation_matrix = data.corr()

    fig = px.imshow(
        correlation_matrix,
        color_continuous_scale=px.colors.diverging.RdBu,
        title="Correlation Matrix"
    )

    fig.show()


"""
This function splits the data into training and testing sets using stratified k-fold cross validation.

Parameters:
    data (pd.DataFrame): The dataset to be split
    n_splits (int): The number of splits to be made

Returns:
    train_data (pd.DataFrame): The training data
    test_data (pd.DataFrame): The testing data

Since having a heartdisease or not is what we want to predict, we use it as the value for y.
"""


def train_test_split_stratified_kfold_cross_validation(data: pd.DataFrame, n_splits=6):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=1)

    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']

    for train_index, test_index in skf.split(X, y):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        yield train_data, test_data


"""
This function trains the model using the training data.

Parameters:
    train_data (pd.DataFrame): The training data
    kernel (str): The kernel to be used in the model

Returns:
    svc_classifier (SVC): The trained model
"""


def train_model(train_data, n_neighbors=5) -> KNeighborsClassifier:
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data.drop('HeartDisease', axis=1))
    y_train = train_data['HeartDisease']

    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(X_train, y_train)

    return knn_classifier


"""
This function evaluates the model using the testing data.

Parameters:
    model (SVC): The trained model
    test_data (pd.DataFrame): The testing data

Returns:
    accurancy (float): The accuracy of the model
"""


def evaluate_model(model, test_data) -> float:
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(test_data.drop('HeartDisease', axis=1))
    y_test = test_data['HeartDisease']

    y_prediction = model.predict(X_test)

    accuracy = roc_auc_score(y_test, y_prediction)

    return accuracy


"""
This function runs the training and testing of the model.

Parameters:
    data (pd.DataFrame): The dataset to be used
    n_splits (int): The number of splits to be made

Returns:
    None
"""


def plot_precision_recall_curve(precision, recall, auc_score):
    fig = px.line(x=recall, y=precision, title="Precision-Recall Curve")
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        shapes=[
            dict(
                type="line",
                x0=0,
                y0=1,
                x1=1,
                y1=0,
                line=dict(color="black", dash="dash"),
            )
        ],
        annotations=[
            dict(
                x=0.5,
                y=0.5,
                xref="x",
                yref="y",
                text=f"AUC={auc_score:.2f}",
                showarrow=False,
                font=dict(size=16),
            )
        ],
    )
    fig.show()


def run_training_and_testing(data, n_splits=6) -> None:
    for fold, (train_data, test_data) in enumerate(
            train_test_split_stratified_kfold_cross_validation(data, n_splits=n_splits)):
        print(f"Fold: {fold + 1}")

        knn_model = train_model(train_data)
        accuracy = evaluate_model(knn_model, test_data)

        print(f"KNN Accuracy: {accuracy}")

        scaler = MinMaxScaler()
        X_test = scaler.fit_transform(test_data.drop('HeartDisease', axis=1))
        y_test = test_data['HeartDisease']

        y_score = knn_model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        pr_auc = auc(recall, precision)

        plot_precision_recall_curve(precision, recall, pr_auc)


def main():
    data = load_dataset(
        path="/Users/karm1616/Desktop/Univeristy/ML-project/datasets/heart.csv")
    transformed_data = transform_columns(data)
    run_training_and_testing(transformed_data, n_splits=6)
    plot_correlation_matrix(transformed_data)


if __name__ == "__main__":
    main()
