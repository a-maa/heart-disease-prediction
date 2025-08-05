import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

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

def plot_correlation_matrix(data: pd.DataFrame) -> None:
    correlation_matrix = data.corr()

    fig = px.imshow(
        correlation_matrix,
        color_continuous_scale=px.colors.diverging.RdBu,
        title="Correlation Matrix"
    )

    fig.show()

def train_test_split_stratified_kfold_cross_validation(data: pd.DataFrame, n_splits=6):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=1)

    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']

    for train_index, test_index in skf.split(X, y): 
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        yield train_data, test_data

def train_model(train_data, kernel='linear') -> SVC:
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data.drop('HeartDisease', axis=1))
    y_train = train_data['HeartDisease']

    svc_classifier = SVC(kernel=kernel, C=1.0, gamma='scale')
    svc_classifier.fit(X_train, y_train)

    return svc_classifier

def evaluate_model(model, test_data) -> float:
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(test_data.drop('HeartDisease', axis=1))
    y_test = test_data['HeartDisease']

    y_prediction = model.predict(X_test)

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
        "sigmoid": {
            "roc_auc": [],
            "precision": [],
            "recall": [],
            "f1": []
        },
        "poly": {
            "roc_auc": [],
            "precision": [],
            "recall": [],
            "f1": []
        },
        "rbf": {
            "roc_auc": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

    }
    for fold, (train_data, test_data) in enumerate(train_test_split_stratified_kfold_cross_validation(data, n_splits=n_splits)):
        print(f"Fold: {fold + 1}")

        linear_svm_model = train_model(train_data, kernel="linear")
        sigmoid_svm_model = train_model(train_data, kernel="sigmoid")
        poly_svm_model = train_model(train_data, kernel="poly")
        rbf_svm_model = train_model(train_data, kernel="rbf")

        linear_roc_auc, linear_precision, linear_recall, linear_f1 = evaluate_model(linear_svm_model, test_data)
        sigmoid_roc_auc, sigmoid_precision, sigmoid_recall, sigmoid_f1 = evaluate_model(sigmoid_svm_model, test_data)
        poly_roc_auc, poly_precision, poly_recall, poly_f1 = evaluate_model(poly_svm_model, test_data)
        rbf_roc_auc, rbf_precision, rbf_recall, rbf_f1 = evaluate_model(rbf_svm_model, test_data)

        results["linear"]["roc_auc"].append(linear_roc_auc)
        results["linear"]["precision"].append(linear_precision)
        results["linear"]["recall"].append(linear_recall)
        results["linear"]["f1"].append(linear_f1)

        results["sigmoid"]["roc_auc"].append(sigmoid_roc_auc)
        results["sigmoid"]["precision"].append(sigmoid_precision)
        results["sigmoid"]["recall"].append(sigmoid_recall)
        results["sigmoid"]["f1"].append(sigmoid_f1)

        results["poly"]["roc_auc"].append(poly_roc_auc)
        results["poly"]["precision"].append(poly_precision)
        results["poly"]["recall"].append(poly_recall)
        results["poly"]["f1"].append(poly_f1)

        results["rbf"]["roc_auc"].append(rbf_roc_auc)
        results["rbf"]["precision"].append(rbf_precision)
        results["rbf"]["recall"].append(rbf_recall)
        results["rbf"]["f1"].append(rbf_f1)

    return results

def calculate_mean_results(results: dict) -> dict:
    mean_result = {
        "linear": {
            "roc_auc": np.mean(results["linear"]["roc_auc"]),
            "precision": np.mean(results["linear"]["precision"]),
            "recall": np.mean(results["linear"]["recall"]),
            "f1": np.mean(results["linear"]["f1"])
        },
        "sigmoid": {
            "roc_auc": np.mean(results["sigmoid"]["roc_auc"]),
            "precision": np.mean(results["sigmoid"]["precision"]),
            "recall": np.mean(results["sigmoid"]["recall"]),
            "f1": np.mean(results["sigmoid"]["f1"])
        },
        "poly": {
            "roc_auc": np.mean(results["poly"]["roc_auc"]),
            "precision": np.mean(results["poly"]["precision"]),
            "recall": np.mean(results["poly"]["recall"]),
            "f1": np.mean(results["poly"]["f1"])
        },
        "rbf": {
            "roc_auc": np.mean(results["rbf"]["roc_auc"]),
            "precision": np.mean(results["rbf"]["precision"]),
            "recall": np.mean(results["rbf"]["recall"]),
            "f1": np.mean(results["rbf"]["f1"])
        }
    }

    return mean_result

def main():
    data = load_dataset()
    transformed_data = transform_columns(data)
    results = run_training_and_testing(transformed_data, n_splits=6)
    mean_results = calculate_mean_results(results)

    print(f"Mean result roc_auc linear: {mean_results['linear']['roc_auc']}")
    print(f"Mean result precision linear: {mean_results['linear']['precision']}")
    print(f"Mean result recall linear: {mean_results['linear']['recall']}")
    print(f"Mean result f1 linear: {mean_results['linear']['f1']}")

    print(f"\nMean result roc_auc sigmoid: {mean_results['sigmoid']['roc_auc']}")
    print(f"Mean result precision sigmoid: {mean_results['sigmoid']['precision']}")
    print(f"Mean result recall sigmoid: {mean_results['sigmoid']['recall']}")
    print(f"Mean result f1 sigmoid: {mean_results['sigmoid']['f1']}")

    print(f"\nMean result roc_auc poly: {mean_results['poly']['roc_auc']}")
    print(f"Mean result precision poly: {mean_results['poly']['precision']}")
    print(f"Mean result recall poly: {mean_results['poly']['recall']}")
    print(f"Mean result f1 poly: {mean_results['poly']['f1']}")

    print(f"\nMean result roc_auc rbf: {mean_results['rbf']['roc_auc']}")
    print(f"Mean result precision rbf: {mean_results['rbf']['precision']}")
    print(f"Mean result recall rbf: {mean_results['rbf']['recall']}")
    print(f"Mean result f1 rbf: {mean_results['rbf']['f1']}")
    # plot_correlation_matrix(transformed_data)

if __name__ == "__main__":
    main()