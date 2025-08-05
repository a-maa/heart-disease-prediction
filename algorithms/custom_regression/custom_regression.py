import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from custom_regression_class import CustomRegression

def load_dataset(type="default", path="./datasets/heart.csv") -> pd.DataFrame:
    data = pd.read_csv(path)

    match type:
        case "str":
            return data.astype(str)
        case "default":
            return data
        case _:
            raise ValueError("Invalid type")

def transform_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop('FastingBS', axis=1)
    data['RestingBP'] = data['RestingBP'].replace(0, data['RestingBP'].median())
    data['Cholesterol'] = data['Cholesterol'].replace(0, data['Cholesterol'].median())
    data['Oldpeak'] = data['Oldpeak'].abs()

    sex_encoded = pd.get_dummies(data['Sex'], prefix='Sex', drop_first=True)
    data = pd.concat([data, sex_encoded], axis=1).drop('Sex', axis=1)
    
    exercise_angina_encoded = pd.get_dummies(data['ExerciseAngina'], prefix='ExerciseAngina', drop_first=True)
    data = pd.concat([data, exercise_angina_encoded], axis=1).drop('ExerciseAngina', axis=1)
    
    chest_pain_type_encoded = pd.get_dummies(data['ChestPainType'], prefix='ChestPainType')
    data = pd.concat([data, chest_pain_type_encoded], axis=1).drop('ChestPainType', axis=1)
    
    resting_ecg_encoded = pd.get_dummies(data['RestingECG'], prefix='RestingECG')
    data = pd.concat([data, resting_ecg_encoded], axis=1).drop('RestingECG', axis=1)
    
    st_slope_encoded = pd.get_dummies(data['ST_Slope'], prefix='ST_Slope')
    data = pd.concat([data, st_slope_encoded], axis=1).drop('ST_Slope', axis=1)

    return data

def train_test_split_stratified_kfold_cross_validation(data: pd.DataFrame, n_splits=6):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=1)

    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']

    for train_index, test_index in skf.split(X, y):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        yield train_data, test_data

def train_models(train_data):
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_data.drop('HeartDisease', axis=1))
    y_train = train_data['HeartDisease']

    clsf = CustomRegression()
    clsf_logistic = CustomRegression(type="logistic")

    clsf.fit(x_train, y_train)
    clsf_logistic.fit(x_train, y_train)

    return clsf, clsf_logistic

def evaluate_model(model, test_data) -> float:
    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(test_data.drop('HeartDisease', axis=1))
    y_test = test_data['HeartDisease']

    y_prediction = model.predict(x_test)

    threshold = 0.5

    y_prediction_binary = np.where(y_prediction > threshold, 1, 0)

    roc_auc = roc_auc_score(y_test, y_prediction)
    precision = precision_score(y_test, y_prediction_binary)
    recall = recall_score(y_test, y_prediction_binary)
    f1 = f1_score(y_test, y_prediction_binary)

    return roc_auc, precision, recall, f1

def run_training_and_testing(data, n_splits=6) -> None:
    results = {
        "linear": {
            "roc_auc": [],
            "precision": [],
            "recall": [],
            "f1": []
        },
        "logistic": {
            "roc_auc": [],
            "precision": [],
            "recall": [],
            "f1": []
        }
    }

    for _, (train_data, test_data) in enumerate(train_test_split_stratified_kfold_cross_validation(data, n_splits=n_splits)):
        clsf, clsf_logistic = train_models(train_data)

        linear_roc_auc, linear_precision, linear_recall, linear_f1 = evaluate_model(clsf, test_data)
        logistic_roc_auc, logistic_precision, logistic_recall, logistic_f1 = evaluate_model(clsf_logistic, test_data)

        results["linear"]["roc_auc"].append(linear_roc_auc)
        results["linear"]["precision"].append(linear_precision)
        results["linear"]["recall"].append(linear_recall)
        results["linear"]["f1"].append(linear_f1)

        results["logistic"]["roc_auc"].append(logistic_roc_auc)
        results["logistic"]["precision"].append(logistic_precision)
        results["logistic"]["recall"].append(logistic_recall)
        results["logistic"]["f1"].append(logistic_f1)


    return results

def calculate_mean_results(results: dict) -> dict:
    mean_results = {
        "linear": {
            "roc_auc": np.mean(results["linear"]["roc_auc"]),
            "precision": np.mean(results["linear"]["precision"]),
            "recall": np.mean(results["linear"]["recall"]),
            "f1": np.mean(results["linear"]["f1"])
        },
        "logistic": {
            "roc_auc": np.mean(results["logistic"]["roc_auc"]),
            "precision": np.mean(results["logistic"]["precision"]),
            "recall": np.mean(results["logistic"]["recall"]),
            "f1": np.mean(results["logistic"]["f1"])
        }
    }

    return mean_results

def main():
    data = load_dataset()
    transformed_data = transform_columns(data)
    results = run_training_and_testing(transformed_data, n_splits=6)
    mean_results = calculate_mean_results(results)

    print("Mean results custom linear roc_auc: ", mean_results['linear']['roc_auc'])
    print("Mean results custom linear precision: ", mean_results['linear']['precision'])
    print("Mean results custom linear recall: ", mean_results['linear']['recall'])
    print("Mean results custom linear f1: ", mean_results['linear']['f1'])

    print("\nMean results custom logistic roc_auc: ", mean_results['logistic']['roc_auc'])
    print("Mean results custom logistic precision: ", mean_results['logistic']['precision'])
    print("Mean results custom logistic recall: ", mean_results['logistic']['recall'])
    print("Mean results custom logistic f1: ", mean_results['logistic']['f1'])

if __name__ == "__main__":
    main()