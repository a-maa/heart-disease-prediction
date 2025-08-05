import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


def load_dataset(type="default", path="./datasets/heart.csv") -> pd.DataFrame:
    data = pd.read_csv(path)
    if type == "str":
        return data.astype(str)
    elif type == "default":
        return data
    else:
        raise ValueError("Invalid type")


def transform_columns(data: pd.DataFrame) -> pd.DataFrame:
    data["Sex"] = data["Sex"].map({'M': 1, 'F': 0})
    data["ExerciseAngina"] = data["ExerciseAngina"].map({'Y': 1, 'N': 0})
    data["ChestPainType"] = data["ChestPainType"].map(
        {'ATA': 1, 'NAP': 2, 'ASY': 3, 'TA': 4})
    data["RestingECG"] = data["RestingECG"].map(
        {'Normal': 0, 'ST': 1, 'LVH': 2})
    data["ST_Slope"] = data["ST_Slope"].map({'Up': 1, 'Flat': 2, 'Down': 3})
    return data


def train_test_split_stratified_kfold_cross_validation(data: pd.DataFrame, n_splits=6):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=1)
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']
    for train_index, test_index in skf.split(X, y):
        yield data.iloc[train_index], data.iloc[test_index]


def train_model(train_data, n_neighbors=25) -> KNeighborsClassifier:
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data.drop('HeartDisease', axis=1))
    y_train = train_data['HeartDisease']
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(X_train, y_train)
    return knn_classifier


def evaluate_model(model, test_data) -> dict:
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(test_data.drop('HeartDisease', axis=1))
    y_test = test_data['HeartDisease']
    y_prediction_proba = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_prediction_binary = np.where(y_prediction_proba > threshold, 1, 0)
    return {
        'roc_auc': roc_auc_score(y_test, y_prediction_proba),
        'precision': precision_score(y_test, y_prediction_binary),
        'recall': recall_score(y_test, y_prediction_binary),
        'f1': f1_score(y_test, y_prediction_binary)
    }


def run_training_and_testing(data, n_splits=6) -> None:
    results = {"roc_auc": [], "precision": [], "recall": [], "f1": []}
    for _, (train_data, test_data) in enumerate(train_test_split_stratified_kfold_cross_validation(data, n_splits=n_splits)):
        knn_model = train_model(train_data)
        metrics = evaluate_model(knn_model, test_data)
        for key, value in metrics.items():
            results[key].append(value)
    for metric in results:
        print(f"Average {metric}: {np.mean(results[metric])}")


def main():
    data = load_dataset(
        path="/Users/karm1616/Desktop/Univeristy/ML-project/datasets/heart.csv")
    transformed_data = transform_columns(data)
    run_training_and_testing(transformed_data, n_splits=6)


if __name__ == "__main__":
    main()
