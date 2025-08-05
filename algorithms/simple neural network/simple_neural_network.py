import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input

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


def train_model(train_data) -> any:
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_data.drop('HeartDisease', axis=1))
    input_dim = x_train.shape[1]

    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(1, activation='tanh', input_dim=11))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    y_train = train_data['HeartDisease']

    history = model.fit(x_train, y_train, epochs=3, batch_size=16)

    return model, history

def evaluate_model(model, test_data) -> float:
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(test_data.drop('HeartDisease', axis=1))
    y_test = test_data['HeartDisease']

    probabilities = model.predict(X_test)

    threshold = 0.5

    binary_predictions = np.where(probabilities > threshold, 1, 0)

    roc_auc = roc_auc_score(y_test, probabilities)
    precision = precision_score(y_test, binary_predictions)
    recall = recall_score(y_test, binary_predictions)
    f1 = f1_score(y_test, binary_predictions)

    return roc_auc, precision, recall, f1

def plot_training_history(history) -> None: 
    fig = px.line(
        history.history,
        y=['loss', 'accuracy'],
        labels={'index': 'epoch', 'value': 'value', 'variable': 'metric'},
        title='Training History'
    )

    fig.show()

def run_training_and_testing(data, n_splits=6) -> None:
    results = {
        "neural network": {
            "roc_auc": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "history": {
                "loss": [],
                "accuracy": []
            }
        },
    }

    for _, (train_data, test_data) in enumerate(train_test_split_stratified_kfold_cross_validation(data, n_splits=n_splits)):

        model, history = train_model(train_data)
        roc_auc, precision, recall, f1 = evaluate_model(model, test_data)

        results["neural network"]["roc_auc"].append(roc_auc)
        results["neural network"]["precision"].append(precision)
        results["neural network"]["recall"].append(recall)
        results["neural network"]["f1"].append(f1)
        results["neural network"]["history"]["loss"].append(history.history['loss'])
        results["neural network"]["history"]["accuracy"].append(history.history['accuracy'])

    return results
        
def calculate_mean_results(results: dict) -> dict:
    mean_results = {
        "neural network": {
            "roc_auc": np.mean(results["neural network"]["roc_auc"]),
            "precision": np.mean(results["neural network"]["precision"]),
            "recall": np.mean(results["neural network"]["recall"]),
            "f1": np.mean(results["neural network"]["f1"]),
            "history": {
                "loss": np.mean(np.array(results["neural network"]["history"]["loss"]).flatten()),
                "accuracy": np.mean(np.array(results["neural network"]["history"]["accuracy"]).flatten())
            }
        }
    }

    return mean_results

def main():
    data = load_dataset()
    transformed_data = transform_columns(data)
    results = run_training_and_testing(transformed_data, n_splits=6)
    mean_results = calculate_mean_results(results)

    print(f"Mean result roc_auc: {mean_results['neural network']['roc_auc']}")
    print(f"Mean result precision: {mean_results['neural network']['precision']}")
    print(f"Mean result recall: {mean_results['neural network']['recall']}")
    print(f"Mean result f1: {mean_results['neural network']['f1']}")

if __name__ == "__main__":
    main()